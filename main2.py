import torch
torch.manual_seed(0)
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from tqdm import tqdm
import torch.nn.utils.prune as prune
import os
import zipfile
from model import Net

"""
def fix_seed(seed):
    # random

    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # Tensorflow


SEED = 42
fix_seed(SEED)
"""



transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

testset=torchvision.datasets.CIFAR10(root='./data',
                                     train=False,
                                     download=True,
                                     transform=transform)

testloader=torch.utils.data.DataLoader(testset,
                                       batch_size=64,
                                       shuffle=False,
                                       num_workers=0)


device=torch.device('cuda'  if torch.cuda.is_available() else 'cpu')

PATH='./cifar10_vgg16.pth.tar'

amount=0.9

model=Net()

pruned_model=model.to(device)
pruned_model.load_state_dict(torch.load(PATH))


parameters_to_prune=(
    (pruned_model.features[0],'weight'),
    (pruned_model.features[3],'weight'),
    (pruned_model.features[6],'weight'),
    (pruned_model.features[11],'weight'),
    (pruned_model.features[14],'weight'),
    (pruned_model.features[19],'weight'),
    (pruned_model.features[22],'weight'),
    (pruned_model.classifier[0],'weight'),
    (pruned_model.classifier[2],'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=amount
)




prune.remove(pruned_model.features[0], 'weight')
prune.remove(pruned_model.features[3], 'weight')
prune.remove(pruned_model.features[6], 'weight')
prune.remove(pruned_model.features[11], 'weight')
prune.remove(pruned_model.features[14], 'weight')
prune.remove(pruned_model.features[19], 'weight')
prune.remove(pruned_model.features[22], 'weight')
prune.remove(pruned_model.classifier[0], 'weight')
prune.remove(pruned_model.classifier[2], 'weight')



correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f'Accuracy of the the Original Model on the 10000 test images: {accuracy:.1f} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = pruned_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f'Accuracy of the the Pruned Model on the 10000 test images: {accuracy:.1f} %')


total_params_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
pruned_model_params_count = sum(torch.nonzero(param).size(0) for param in pruned_model.parameters() if param.requires_grad)

print(f'Original Model parameter count: {total_params_count:,}')
print(f'Pruned Model parameter count: {pruned_model_params_count:,}')
print(f'Compressed Percentage: {(100 - (pruned_model_params_count / total_params_count) * 100):.2f}%')






with zipfile.ZipFile('cifar10_vgg16.pth.tar.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write( 'cifar10_vgg16.pth.tar')

original_model_size = os.path.getsize('cifar10_vgg16.pth.tar.zip')

torch.save(pruned_model.state_dict(), "pruned_cifar10_vgg16.pth.tar")

with zipfile.ZipFile('pruned_cifar10_vgg16.pth.tar.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write( 'pruned_cifar10_vgg16.pth.tar')

pruned_model_size = os.path.getsize('pruned_cifar10_vgg16.pth.tar.zip')

print(f'Size of the the Original Model: {original_model_size:,} bytes')
print(f'Size of the the Pruned Model: {pruned_model_size:,} bytes')





