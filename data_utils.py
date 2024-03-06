import numpy as np
import torchvision.transforms.functional as TorchVisionFunc
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Resize

class RotationTransform:
	def __init__(self, angle):
		self.angle = angle

	def __call__(self, x):
		return TorchVisionFunc.rotate(x, self.angle, fill=(0,))


def get_rotated_gtsrb(task_id, shuffle=False, batch_size=32):    
    rotation_degree = task_id * 22.5
    transforms = Compose([
        Resize((32, 32)),
		RotationTransform(rotation_degree),
		ToTensor(),
        Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
		])

    train_loader = DataLoader(torchvision.datasets.GTSRB('./data/', split='train', download=True, transform=transforms), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(torchvision.datasets.GTSRB('./data/', split='test', download=True, transform=transforms),  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def get_rotated_svhn(task_id, shuffle=False, batch_size=32):    
    rotation_degree = task_id * 22.5
    transforms = Compose([
		RotationTransform(rotation_degree),
		ToTensor(),
        Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
		])

    train_loader = DataLoader(torchvision.datasets.SVHN('./data/', split='train', download=True, transform=transforms), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(torchvision.datasets.SVHN('./data/', split='test', download=True, transform=transforms),  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def get_rotated_mnist(task_id, shuffle=False, batch_size=32):    
    rotation_degree = task_id * 22.5
    transforms = Compose([
		RotationTransform(rotation_degree),
		ToTensor(),
		])

    train_loader = DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def get_rotated_fashion_mnist(task_id, batch_size=32):    
    rotation_degree = task_id * 22.5
    transforms = Compose([
		RotationTransform(rotation_degree),
		ToTensor(),
		])

    train_loader = DataLoader(torchvision.datasets.FashionMNIST('./data/', train=True, download=True, transform=transforms), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(torchvision.datasets.FashionMNIST('./data/', train=False, download=True, transform=transforms),  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def get_continual_learning_tasks(task_name, num_tasks=5, batch_size=32):
    datasets = {}
    
    if task_name == 'mnist':
        f = get_rotated_mnist
    elif task_name == 'fashion-mnist':
        f = get_rotated_fashion_mnist
    elif task_name == 'svhn':
        f = get_rotated_svhn
    elif task_name == 'gtsrb':
        f = get_rotated_gtsrb

    for task_id in range(0, num_tasks):
        train_loader, test_loader = f(task_id, batch_size)
        datasets[task_id] = {'train': train_loader, 'test': test_loader}
    return datasets
