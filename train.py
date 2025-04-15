import copy

import torch
import yaml
import matplotlib.pyplot as plt
from server.server import Server
from client.client import Client
from torch.utils.data import DataLoader
import time

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='training.log',
    filemode='a'
)

def main():
    # 加载配置
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    server = Server(config["num_clients"], config["num_classes"])
    device = next(server.global_model.parameters()).device

    test_data = torch.load(config["test_folder"], weights_only=False)
    test_loader = DataLoader(
        [(torch.tensor(img).to(device), torch.tensor(label).to(device)) for img, label in test_data],
        batch_size=64
    )

    client_loss_history = []
    server_acc_history = []

    for round in range(config["num_rounds"]):
        logging.info(f"Round {round + 1}/{config['num_rounds']}")
        client_weights = []
        round_losses = []
        for client_id in range(config["num_clients"]):
            client = Client(client_id, config["data_folder"], config["batch_size"])
            weights, losses = client.train(copy.deepcopy(server.global_model),
                                           config["local_epochs"],
                                           config["num_classes"],
                                           config["num_weak_aug_rounds"]
                                           )
            client_weights.append(weights)
            round_losses.extend(losses)

        server.aggregate(client_weights)
        accuracy = server.evaluate(test_loader)
        logging.info(f"Global Accuracy: {accuracy*100:.2f}")

        client_loss_history.append(sum(round_losses) / len(round_losses))
        server_acc_history.append(accuracy)

    with open("client_loss_history.pkl", "wb") as f:
        torch.save(client_loss_history, f)
    with open("server_acc_history.pkl", "wb") as f:
        torch.save(server_acc_history, f)


if __name__ == "__main__":
    main()
