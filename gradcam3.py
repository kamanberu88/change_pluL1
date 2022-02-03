from PIL import Image

import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import urllib
import pickle
import matplotlib.pyplot as plt
from model import Net
import torch.nn.utils.prune as prune
from class_gradcam import GradCam

classes=np.array(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])
transform=transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),


])
PATH='./cifar10_vgg16.pth.tar'


for i in range(0,10,1):
    amount=(i/10)
    print(amount)

    model=Net()
    pruned_model = model
    pruned_model.load_state_dict(torch.load(PATH))

    parameters_to_prune = (
        (pruned_model.features[0], 'weight'),
        (pruned_model.features[3], 'weight'),
        (pruned_model.features[6], 'weight'),
        (pruned_model.features[11], 'weight'),
        (pruned_model.features[14], 'weight'),
        (pruned_model.features[19], 'weight'),
        (pruned_model.features[22], 'weight'),
        (pruned_model.classifier[0], 'weight'),
        (pruned_model.classifier[2], 'weight'),
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

    test_image = Image.open("./test_09833.png")
    test_image_tensor = (transform((test_image))).unsqueeze(dim=0)

    image_size = test_image.size
    print("image size:", image_size)

    plt.imshow(test_image)
    grad_cam = GradCam(pruned_model)
    feature_image = grad_cam(test_image_tensor).squeeze(dim=0)
    feature_image = transforms.ToPILImage()(feature_image)
    outputs=pruned_model(test_image_tensor)
    pred_idx = pruned_model(test_image_tensor).max(1)[1]
    print("predicted: ", classes[int(pred_idx)])
    print('probability:',torch.max(F.softmax(outputs,1)).item())
    #plt.title("Grad-

    plt.imshow(feature_image.resize(image_size))
    plt.axis("off")
    plt.savefig('./horse/test_9833/figure083-{}.jpg'.format(amount),bbox_inches='tight',pad_inches = 0)
    print("############################################################")




