import sys
from getopt import getopt
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from model_zoo import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


if __name__ == '__main__':
    model = MyCNN().to(device)
    load_path = 'model/default_model.kpl'
    opts, args = getopt(sys.argv[1:], 'm:l:',
                        ['model=', 'load_path='])
    for opt_name, opt_value in opts:
        if opt_name in ('-m', '--model'):
            if opt_value == 'ResNet':
                model = ResNet(BasicBlock, [2, 2, 2, 2], 10).to(device)
            elif opt_value == 'CNN':
                model = MyCNN().to(device)
            else:
                print("Error model name, exit program")
                exit(-1)
        elif opt_name in ('-l', '--load_path'):
            load_path = opt_value

    dic = torch.load(load_path, map_location=device)
    model.load_state_dict(dic)
    test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
    test_loss, correct = 0, 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()

    print(f"Test Error:\n Accuracy : {(100 * correct / len(test_dataset)):>0.1f}%")
    print(test_loss / len(test_loader))
