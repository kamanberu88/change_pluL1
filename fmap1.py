import torch

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
from layer import LayerActivations
from PIL import Image
from torch.autograd import Variable


classes=np.array(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])

transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

testset=torchvision.datasets.CIFAR10(root='./data',
                                     train=False,
                                     download=True,
                                     transform=transform)

testloader=torch.utils.data.DataLoader(testset,
                                       batch_size=32,
                                       shuffle=False,
                                       num_workers=0)


device=torch.device('cuda'  if torch.cuda.is_available() else 'cpu')

PATH='./cifar10_vgg16.pth.tar'

amount=0.9

img=Image.open("./train_00083.png")
plt.imshow(img)
plt.show()
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

inputs=transform(img)
inputs=inputs.unsqueeze(0).to(device)
print(inputs.shape)

"""
inputs=transform(img)
inputs=inputs.unsqueeze(0).to(device)
print(inputs.shape)

model=Net().to(device)
print(model)
model.eval()


outputs=model(inputs)
_,predicted=torch.max(outputs,1)
print('predicted_label',('%5s' % classes[predicted]))
print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')

inputs=transform(img)
inputs=inputs.unsqueeze(0).to(device)
print(inputs.shape)
"""


pruned_model.eval()

outputs=pruned_model(inputs)
_,predicted=torch.max(outputs,1)
print('predicted_label',('%5s' % classes[predicted]))

conv_out=LayerActivations(pruned_model.features,24)
o=model(Variable(inputs.cuda()))
conv_out.remove()
act=conv_out.features
print(act.shape)


fig=plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0,right=1,bottom=0,top=0.8,hspace=0,wspace=0.2)
for i in range(5):
    ax=fig.add_subplot(12,5,i+1,xticks=[],yticks=[])
    ax.imshow(act[0][i].detach().numpy())
#plt.show()
fig.savefig('./result_fmap/83/pruning-amount-{}.png'.format(amount))

