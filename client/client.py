import torch
import copy
from torch.utils.data import DataLoader
from torchvision import transforms

from client.dataset import FederatedDataset
from models.semisup import SemiSupervised


class Client:
    def __init__(self, client_id, data_dir):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "")
        self.train_loss = []

        dataset = FederatedDataset(
            f"{data_dir}/train/labeled/data.pt",
            f"{data_dir}/train/unlabeled/data.pt"
        )
        self.loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # 增强策略
        # 随机水平翻转（50%概率）
        # 随机缩放裁剪（保留60%-100%区域）
        # 颜色抖动（亮度/对比度/饱和度变化40%，色调变化10%）
        # 随机灰度化（20%概率）
        # 高斯模糊（σ范围0.1-0.5）
        self.strong_aug = transforms.Compose([
            transforms.Lambda(lambda x: x.cpu()),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=32, scale=(0.6, 1.0)),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.Lambda(lambda x: x.to(self.device))
        ])

    def train(self, global_model, epochs, lr=0.001):
        model = copy.deepcopy(global_model).to(self.device)
        # L2正则防止客户端过拟合
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # 学习率递减
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: min(e / 5.0, 1.0))
        model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for images, labels in self.loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                labeled_mask = (labels != -1)
                unlabeled_mask = (labels == -1)

                loss_sup = torch.tensor(0.0, device=self.device, requires_grad=True)
                loss_unsup = torch.tensor(0.0, device=self.device, requires_grad=True)

                if torch.any(labeled_mask):
                    labeled_images = images[labeled_mask]
                    labeled_labels = labels[labeled_mask]
                    logits = model(labeled_images)
                    loss_sup = torch.nn.functional.cross_entropy(logits, labeled_labels)

                if torch.any(unlabeled_mask):
                    unlabeled_images = images[unlabeled_mask]
                    with torch.no_grad():
                        pseudo_labels, filtered_data = SemiSupervised.generate_pseudo_labels(model,
                                                                                             unlabeled_images,
                                                                                             epoch=epoch,
                                                                                             total_epochs=epochs)
                        if filtered_data.size(0) > 0:
                            # # 确保数据维度正确 [C, H, W]
                            # if filtered_data.shape[1] != 3:
                            #     filtered_data = filtered_data.permute(0, 3, 1, 2)

                            strong_augmented = self.strong_aug(filtered_data)
                            logits_weak = model(filtered_data)
                            logits_strong = model(strong_augmented)
                            loss_unsup = SemiSupervised.consistency_loss(logits_weak, logits_strong.detach())

                # unsupervised_weight = 0.5 * (1 - epoch / epochs)
                unsupervised_weight = 0.5 * (0.5 + 0.5 * (1 - epoch / epochs))
                total_loss = loss_sup + unsupervised_weight * loss_unsup
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                epoch_loss += total_loss.item()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            self.train_loss.append(epoch_loss / len(self.loader))
            scheduler.step()

        return model.state_dict(), self.train_loss

    def cleanup(self):
        del self.loader.dataset
        del self.loader
        if hasattr(self, 'model'):
            del self.model