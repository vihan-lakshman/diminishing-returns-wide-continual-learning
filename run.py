import argparse
import logging
import numpy as np
import torch
import torch.nn as nn

from data_utils import get_continual_learning_tasks
from model import MLP


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--task_name", choices=['mnist', 'fashion-mnist', 'svhn', 'gtsrb'], required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
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


def get_mlp_model(task_name, num_layers, width):

    if task_name == 'gtsrb':
        num_labels = 43
    else:
        num_labels = 10

    if task_name in ['mnist', 'fashion-mnist']:
        input_dim = 784
    else:
        input_dim = 3072
    
    model = MLP(input_dim, num_labels, width, num_layers)

    return model



def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()                               # count number of correct ones


def train(data_loader, model, device="cpu"):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
    total_loss = 0
    total_correct = 0
    criterion = nn.CrossEntropyLoss()
    
    for data, target in data_loader:
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


def test(test_loader, model, device):
    model.eval()
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0
    criterion = nn.CrossEntropyLoss()
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


def train_continual(datasets, model, device):
    epochs = 5
    running_test_accs = {i: [] for i in range(5)}
    learning_accs = []
    for task_id in range(5):
        train_dataloader = datasets[task_id]['train']
        for epoch in range(epochs):
            train(train_dataloader, model, device)
        
        for test_task_id in range(5):   
            if test_task_id > task_id:
                test_acc = 0
            else:
                test_acc = test(datasets[test_task_id]['test'], model, device)
		    
            if test_task_id == task_id:
                     learning_accs.append(test_acc)
                    
            
            running_test_accs[test_task_id].append(test_acc)

        
    model.eval()

    score = np.mean([running_test_accs[i][-1] for i in running_test_accs.keys()])
    forget = np.mean([max(running_test_accs[i])-running_test_accs[i][-1] for i in range(5)])
    learning_acc = np.mean(learning_accs)

    return score, forget, learning_acc  

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = set_device()
    datasets = get_continual_learning_tasks(args.task_name)
    model = get_mlp_model(args.task_name, args.num_layers, args.width)
    model.to(device)
    logging.basicConfig(filename=f'{args.task_name}_{args.num_layers}l_{args.width}w.log',filemode='a', encoding='utf-8', level=logging.INFO)

    average_accuracy, average_forgetting, learning_accuracy = train_continual(datasets, model, device)
    print(average_accuracy, average_forgetting)

    logging.info(f"Width {args.width}: "
                 f"Accuracy: {average_accuracy}, "
                 f"Forgetting: {average_forgetting}, "
                 f"Learning Acc: {learning_accuracy}")


if __name__=='__main__':
    main()
