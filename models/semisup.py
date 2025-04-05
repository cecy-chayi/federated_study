import torch
import torchvision.transforms as T

class SemiSupervised:
    @staticmethod
    def generate_pseudo_labels(model, unlabeled_data, temperature=0.5, epoch=0, total_epochs=20):
        # 置信度阈值过滤
        model.eval()
        with torch.no_grad():
            threshold = 0.75 + 0.2 * (1 - epoch / total_epochs)
            weak_aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(size=32, padding=4)
            ])
            strong_aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(size=32, padding=4),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomGrayscale(p=0.2)
            ])

            weak_view = weak_aug(unlabeled_data)
            strong_view = strong_aug(unlabeled_data)

            logits_weak = model(weak_view)
            logits_strong = model(strong_view)
            probs = (torch.softmax(logits_weak/temperature, dim=1) + torch.softmax(logits_strong/temperature, dim=1)) / 2

            max_probs, pseudo_labels = torch.max(probs, dim=1)

            # 类别平衡过滤（防止只选择特定类别）
            class_counts = torch.bincount(pseudo_labels, minlength=probs.size(1))
            class_weights = 1. / (class_counts + 1e-4)
            sample_weights = class_weights[pseudo_labels]

            # mask = (max_probs > threshold) & (sample_weights < torch.quantile(sample_weights, 0.9))
            mask = (max_probs > threshold) & (sample_weights < torch.quantile(sample_weights, 0.95))

            return pseudo_labels[mask], unlabeled_data[mask]

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
