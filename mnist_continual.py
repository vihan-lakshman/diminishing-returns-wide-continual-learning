import argparse
import logging
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms.functional as TorchVisionFunc

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose


class RotationTransform:
	def __init__(self, angle):
		self.angle = angle

	def __call__(self, x):
		return TorchVisionFunc.rotate(x, self.angle, fill=(0,))


class SimpleMLP(nn.Module):
    def __init__(self, num_layers=1):
        super().__init__()
        layers = [nn.Flatten()]
        
        for _ in range(num_layers):
            layers += [nn.Linear(width, width), nn.ReLU()]

        layers.append(nn.Linear(width, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args


def set_device():
    print('Using PyTorch version:', torch.__version__)
    if torch.cuda.is_available():
        print('Using GPU, device name:', torch.cuda.get_device_name(0))
        device = torch.device('cuda')
    else:
        print('No GPU found, using CPU instead.') 
        device = torch.device('cpu')
    
    return device

def get_rotated_mnist(task_id, shuffle=False, batch_size=32):    
    rotation_degree = task_id * 22.5
    transforms = Compose([
		RotationTransform(rotation_degree),
		ToTensor(),
		])

    train_loader = DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def get_rotated_mnist_tasks(num_tasks=5, shuffle=False, batch_size=32):
	datasets = {}
	for task_id in range(0, num_tasks):
		train_loader, test_loader = get_rotated_mnist(task_id, shuffle, batch_size)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets


def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()                               # count number of correct ones


def train(data_loader):
    model.train()
    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
    total_loss = 0
    total_correct = 0
    for data, target in tqdm(data_loader):
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        # Do a forward pass
        output = model(data)
                                                                                        
        # Calculate the loss
        loss = criterion(output, target)
        total_loss += loss
        
        total_correct += correct(output, target)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(test_loader):
    model.eval()
    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(device)
            target = target.to(device)

            # Do a forward pass
            output = model(data)
            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()
                                                                            
            # Count number of correct digits
            total_correct += correct(output, target)

    accuracy = total_correct/num_items
    
    return accuracy

if __name__=='__main__':
    args = parse_args()
    device = set_device()
    datasets = get_rotated_mnist_tasks()

    width = args.width

    logging.basicConfig(filename=f'mnist_{args.num_layers}l_5runs.log',filemode='a', encoding='utf-8', level=logging.INFO)
    
    final_scores = []
    final_forgetting = []
    final_learning_acc = []
    for seed in range(1):
        torch.manual_seed(seed)
        datasets = get_rotated_mnist_tasks()
        model = SimpleMLP(args.num_layers).to(device)

        criterion = nn.CrossEntropyLoss()


        model.train()
        epochs = 5

        running_test_accs = {i: [] for i in range(5)}
        learning_accs = []
        for task_id in range(5):
            train_dataloader= datasets[task_id]['train']
            for epoch in range(epochs):
                train(train_dataloader) # , model, criterion, optimizer)
        
                for test_task_id in range(5):   
                    if test_task_id > task_id:
                        test_acc = 0 # left-padding with zero (to have square matrix)
                    else:
                        test_acc = test(datasets[test_task_id]['test'])
		    
                    if test_task_id == task_id:
                        learning_accs.append(test_acc)
                    
            
                    running_test_accs[test_task_id].append(test_acc)

        
        model.eval()

        score = np.mean([running_test_accs[i][-1] for i in running_test_accs.keys()])
        forget = np.mean([max(running_test_accs[i])-running_test_accs[i][-1] for i in range(5)])
        learning_acc = np.mean(learning_accs)
        
        final_scores.append(score)
        final_forgetting.append(forget)
        final_learning_acc.append(learning_acc)

    logging.info(f"Width {args.width}: 
                 Accuracy: {(np.mean(final_scores), np.std(final_scores))}, 
                 Forgetting: {(np.mean(final_forgetting), np.std(final_forgetting))}, 
                 Learning Acc: {(np.mean(final_learning_acc), np.std(final_learning_acc))}")
        
