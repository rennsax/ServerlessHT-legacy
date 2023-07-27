# pytorch lenet
import argparse

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

EPOCH = 20


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


if __name__ == "__main__":
    torch.random.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g", "--gpu", type=int, required=True, help="which gpu to use", choices=(0, 1)
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, required=True, help="batch size for dataloader"
    )
    parser.add_argument(
        "-l", "--learning-rate", type=float, required=True, help="learning rate"
    )
    parser.add_argument(
        "-m", "--momentum", type=float, required=True, help="momentum for optimizer"
    )
    parser.add_argument("--output", type=str, required=True, help="the file to output")
    args = parser.parse_args()
    print(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    loss_function = nn.CrossEntropyLoss()
    train_set = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=False,
        transform=torchvision.transforms.ToTensor(),
    )
    test_set = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=False,
        transform=torchvision.transforms.ToTensor(),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    optimizer = optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )

    start_time = time.time()
    for epoch in range(1, EPOCH + 1):
        model.train()
        for i, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            optimizer.zero_grad()
            output = model(train_x)
            loss = loss_function(output, train_label)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, step {i}, loss: {loss.item():.3f}")

        total_correct_cnt = 0
        total_sample_cnt = 0

        model.eval()
        for i, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            output = model(test_x)
            pred = output.argmax(dim=1)
            total_correct_cnt += (pred == test_label).sum().item()
            total_sample_cnt += test_x.shape[0]

        acc = total_correct_cnt / total_sample_cnt
        print(f"Epoch {epoch}, test acc: {acc:.3f}\n", flush=True)
        stop_time = time.time()

        if epoch == EPOCH:
            with open(args.output, "a") as f:
                f.write(
                    f"{args.batch_size:d},{args.learning_rate:.4f},{args.momentum:.1f},{acc:.4f},{stop_time-start_time:.2f}\n"
                )
