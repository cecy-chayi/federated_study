import torch
import copy
from torch.utils.data import DataLoader
from torchvision import transforms

from client.dataset import FederatedDataset
from models.semisup import SemiSupervised
import torch.nn.functional as F
import logging


class Client:
    def __init__(self, client_id, data_dir, batch_size):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "")
        self.train_loss = []

        dataset = FederatedDataset(
            f"{data_dir}/train/labeled/data.pt",
            f"{data_dir}/train/unlabeled/data.pt"
        )
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 增强策略
        # 随机水平翻转（50%概率）
        # 随机缩放裁剪（保留60%-100%区域）
        # 颜色抖动（亮度/对比度/饱和度变化40%，色调变化10%）
        # 随机灰度化（20%概率）
        # 高斯模糊（σ范围0.1-0.5）
        self.strong_aug = transforms.Compose([
            transforms.Lambda(lambda x: (x * 255).to(torch.uint8).cpu()),
            transforms.RandAugment(
                num_ops=3,  # 随机选择3个增强操作
                magnitude=9,  # 增强强度设为9（范围0-10）
                num_magnitude_bins=11,  # 默认的magnitude bins数量
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.Lambda(lambda x: x.float().div(255).to(self.device))
        ])

    def train(self, global_model, epochs, num_classes, num_weak_aug_rounds, lr=0.001):
        model = copy.deepcopy(global_model).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: min(e / epochs, 1.0))
        model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for images, labels in self.loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                labeled_mask = (labels != -1)
                unlabeled_mask = (labels == -1)

                loss_sup = None
                loss_unsup = None

                # 有监督损失
                if torch.any(labeled_mask):
                    labeled_images = images[labeled_mask]
                    labeled_labels = labels[labeled_mask]
                    logits = model(labeled_images)
                    loss_sup = F.cross_entropy(logits, labeled_labels)

                # 无监督损失
                if torch.any(unlabeled_mask):
                    unlabeled_images = images[unlabeled_mask]

                    with torch.no_grad():
                        pseudo_labels, filtered_data = SemiSupervised.generate_pseudo_labels(
                            model=model,
                            unlabeled_datas=unlabeled_images,
                            num_weak_aug_rounds=num_weak_aug_rounds,
                            num_classes=num_classes,
                            epoch=epoch,
                            total_epochs=epochs
                        )

                    pseudo_count = filtered_data.size(0)
                    logging.info(f"[Client {self.client_id}][Epoch {epoch}] Pseudo labels used: {pseudo_count}")

                    if pseudo_count > 0:
                        # 修复 RandAugment 输入类型问题
                        strong_augmented = torch.stack([
                            self.strong_aug((img * 255).to(torch.uint8)).float() / 255.0
                            for img in filtered_data
                        ]).to(self.device)

                        logits_strong = model(strong_augmented)
                        loss_unsup = F.cross_entropy(logits_strong, pseudo_labels)

                # 计算总损失
                losses = []
                if loss_sup is not None:
                    losses.append(loss_sup)
                if loss_unsup is not None:
                    losses.append(0.5 * loss_unsup)

                if len(losses) == 0:
                    logging.warning(f"[Client {self.client_id}][Epoch {epoch}] No valid loss. Skipping batch.")
                    continue

                total_loss = sum(losses)

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                epoch_loss += total_loss.item()

            self.train_loss.append(epoch_loss / len(self.loader))
            scheduler.step()

        return model.state_dict(), self.train_loss
