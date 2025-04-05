import torch
from torch.utils.data import Dataset


class FederatedDataset(Dataset):
    def __init__(self, labeled_path, unlabeled_path):
        self.labeled_data = torch.load(labeled_path, weights_only=False)
        self.unlabeled_data = torch.load(unlabeled_path, weights_only=False)

    def __len__(self):
        return len(self.labeled_data) + len(self.unlabeled_data)

    def __getitem__(self, idx):
        if idx < len(self.labeled_data):
            img, label = self.labeled_data[idx]
            return torch.tensor(img), torch.tensor(label)
        else:
            img = self.unlabeled_data[idx - len(self.labeled_data)]
            return torch.tensor(img), torch.tensor(-1)  # -1 无标签
