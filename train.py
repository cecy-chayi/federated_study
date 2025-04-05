import copy

import torch
import yaml
import matplotlib.pyplot as plt
from server.server import Server
from client.client import Client
from torch.utils.data import DataLoader
import time
import psutil

def main():
    # 加载配置
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    server = Server(config["num_clients"])
    device = next(server.global_model.parameters()).device

    test_data = torch.load("./data/cifar10/test/data.pt", weights_only=False)
    test_loader = DataLoader(
        [(torch.tensor(img).to(device), torch.tensor(label).to(device)) for img, label in test_data],
        batch_size=64
    )

    client_loss_history = []
    server_acc_history = []

    for round in range(config["num_rounds"]):
        process = psutil.Process()
        mem = psutil.virtual_memory()
        mem_before = process.memory_info().rss / 1024 / 1024
        print(f"Round {round + 1}/{config['num_rounds']}")
        client_weights = []
        round_losses = []
        for client_id in range(config["num_clients"]):
            client = Client(client_id, "./data/cifar10")
            weights, losses = client.train(copy.deepcopy(server.global_model), config["local_epochs"])
            client_weights.append(weights)
            round_losses.extend(losses)
            client.cleanup()
            del client
            import gc
            gc.collect()

        server.aggregate(client_weights)
        accuracy = server.evaluate(test_loader)
        print(f"Global Accuracy: {accuracy*100:.2f}")

        client_loss_history.append(sum(round_losses) / len(round_losses))
        server_acc_history.append(accuracy)
        mem_after = process.memory_info().rss / 1024 / 1024
        print(f"2 - Round {round} 内存增量: {mem_after - mem_before:.2f}MB, 可用内存： {mem.available / (1024 ** 3):.2f}GB")
        # PyTorch显存监控
        if torch.cuda.is_available():
            print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1e6}MB")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(client_loss_history, 'b-', label="Training Loss")
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.title("Client Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(server_acc_history, 'r-', label="Validation Accuracy")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("Server Validation Accuracy")
    plt.grid(True)

    plt.tight_layout()

    png_name = f'training_metrics_{int(time.time())}.png'
    plt.savefig(png_name)
    plt.close()


if __name__ == "__main__":
    main()
