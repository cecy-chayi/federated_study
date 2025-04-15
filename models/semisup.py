import torch
import torchvision.transforms as T

class SemiSupervised:
    @staticmethod
    def generate_pseudo_labels(model, unlabeled_data, K, temperature=0.5, epoch=0, total_epochs=20):
        # 置信度阈值过滤
        model.eval()
        with torch.no_grad():
            threshold = 0.7 + 0.2 * (1 - epoch / total_epochs)
            weak_aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(size=32, padding=4)
            ])

            # 对每次弱增强，计算其概率分布
            # 最后通过加权平均得到其概率分布
            # 选择最大的概率作为其伪标签
            weak_view = unlabeled_data
            batch_size = unlabeled_data.size(0)
            total_probs = torch.zeros(batch_size, model.num_classes).to(unlabeled_data.device)

            for i in range(K):
                weak_view = weak_aug(weak_view)
                logits = model(weak_view)
                probs = torch.softmax(logits, dim=1)
                total_probs += 2 * (K - i + 1) / (K * (K + 1)) * probs

            max_probs, pseudo_labels = torch.max(total_probs, dim=1)
            mask = max_probs >= threshold
            return pseudo_labels, mask

    @staticmethod
    def consistency_loss(logits1, logits2, method='KL'):
        if method == 'KL':
            probs1 = torch.softmax(logits1, dim=-1)
            probs2 = torch.softmax(logits2, dim=-1)
            return torch.nn.functional.kl_div(probs1.log(), probs2, reduction="batchmean")
        elif method == 'MSE':
            return torch.nn.functional.mse_loss(logits1, logits2)
        elif method == 'CE':
            return torch.nn.functional.cross_entropy(logits1, logits2.argmax(dim=-1))
