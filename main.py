import torch
from sklearn.metrics import accuracy_score
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


transform_train=transforms.Compose(
    [transforms.RandomCrop(32,padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),

     ])

transform_val=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),

     ])




trainset=torchvision.datasets.CIFAR10(root='./data',
                                      train=True,
                                      download=True,
                                      transform=transform_train)

trainloader=torch.utils.data.DataLoader(trainset,
                                        batch_size=64,
                                        shuffle=True,
                                        num_workers=0)

testset=torchvision.datasets.CIFAR10(root='./data',
                                     train=False,
                                     download=True,
                                     transform=transform_val)

testloader=torch.utils.data.DataLoader(testset,
                                       batch_size=64,
                                       shuffle=False,
                                       num_workers=0)


classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
device=torch.device('cuda'  if torch.cuda.is_available() else 'cpu')

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter=iter(trainloader)
images,labels=dataiter.next()

#imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


model=Net()
model=model.to(device)
print(device)
print(model)
criterion=torch.nn.CrossEntropyLoss()
#optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)


scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
    threshold=0.0001,
    verbose=True
)

def train_step(x,t):
    model.train()
    preds=model(x)
    loss=criterion(preds,t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss,preds

def test_step(x,t):
    model.eval()
    preds=model(x)
    loss=criterion(preds,t)

    return loss,preds


epochs=50
history={'loss':[],'accuracy':[],'test_loss':[],'test_accuracy':[]}



for epoch in range(epochs):
    train_loss=0.
    train_acc=0.
    test_loss=0.
    test_acc=0.


    for(inputs,labels) in tqdm(trainloader):

        inputs,labels=inputs.to(device),labels.to(device)
        loss,preds=train_step(inputs,labels)
        train_loss+=loss.item()
        train_acc += accuracy_score(
            labels.tolist(),
            preds.argmax(dim=-1).tolist()
        )

    for (inputs,labels) in tqdm(testloader):

        inputs, labels = inputs.to(device), labels.to(device)
        loss, preds = test_step(inputs, labels)
        test_loss += loss.item()
        test_acc += accuracy_score(
            labels.tolist(),
            preds.argmax(dim=-1).tolist()
        )


    #1epoechにおけるロスと精度
    avg_train_loss = train_loss / len(trainloader)
    avg_train_acc = train_acc / len(trainloader)
    avg_test_loss = test_loss / len(testloader)
    avg_test_acc = test_acc / len(testloader)

    history['loss'].append(avg_train_loss)
    history['accuracy'].append(avg_train_acc)
    history['test_loss'].append(avg_test_loss)
    history['test_accuracy'].append(avg_test_acc)



    if (epoch + 1) % 1 == 0:
        print(
            "epoch{} train_loss:{:.4} train_acc:{:.4} val_loss:{:.4} val_acc:{:.4}"
                .format(
                epoch + 1,
                avg_train_loss,
                avg_train_acc,
                avg_test_loss,
                avg_test_acc
            ))

    scheduler.step(avg_test_acc)

#PATH='./cifar10_vgg16.pth.tar'
#torch.save(model.state_dict(),PATH)




plt.plot(history['loss'],
         marker='.',
         label='loss(Training)')

plt.plot(history['test_loss'],
         marker='.',
         label='loss(Test)')

plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



plt.plot(history['accuracy'],
         marker='.',
         label='accuracy(Training)')

plt.plot(history['test_accuracy'],
         marker='.',
         label='accuracy(Test)')

plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()







