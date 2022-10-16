import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import AutoAugment, ToTensor, AutoAugmentPolicy, Lambda
from model_zoo import *
from getopt import getopt
import sys
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

if __name__ == '__main__':
    model = MyCNN().to(device)

    learning_rate = 1e-3
    epochs = 20
    save_path = 'model/default_model.kpl'
    opts, args = getopt(sys.argv[1:], 'e:m:l:s:',
                        ['epochs=', 'model=', 'learning_rate=',
                         'save_path='])
    for opt_name, opt_value in opts:
        if opt_name in ('-e', '--epochs'):
            epochs = int(opt_value)
        elif opt_name in ('-m', '--model'):
            if opt_value == 'ResNet':
                model = ResNet(BasicBlock, [2, 2, 2, 2], 10).to(device)
            elif opt_value == 'CNN':
                model = MyCNN().to(device)
            else:
                print("Error model name, exit program")
                exit(-1)
        elif opt_name in ('-l', '--learning_rate'):
            learning_rate = float(opt_value)
        elif opt_name in ('-s', '--save_path'):
            save_path = opt_value

    train_transform = transforms.Compose([
        AutoAugment(AutoAugmentPolicy.CIFAR10),
        ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform,
                                                 target_transform=Lambda(
                                                     lambda y: torch.zeros(10, dtype=torch.float).scatter_(0,
                                                                                                           torch.tensor(
                                                                                                               y),
                                                                                                           value=1)))

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    size = len(train_loader.dataset)
    for i in range(1, epochs + 1):
        print(f"Epoch {i}\n----------------------------")
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        if i == 10:
            torch.save(model.state_dict(), f"CNN_model_{i}.kpl")

    torch.save(model.state_dict(), save_path)
