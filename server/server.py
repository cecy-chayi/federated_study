import torch
from models.cnn import CNN
from torchvision.models import resnet18


class Server:
    def __init__(self, num_clients, num_classes):
        self.global_model = resnet18(num_classes=num_classes)
        self.global_model.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.global_model.maxpool = torch.nn.Identity()
        self.num_clients = num_clients

    def aggregate(self, client_weights):
        avg_weights = {}
        for key in client_weights[0].keys():
            avg_weights[key] = torch.mean(
                torch.stack([w[key].float() for w in client_weights]), dim=0
            )
        self.global_model.load_state_dict(avg_weights)

    def evaluate(self, test_loader):
        device = next(self.global_model.parameters()).device
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total
