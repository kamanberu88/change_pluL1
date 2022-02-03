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

class GradCam:
    def __init__(self,model):
        self.model=model.eval()
        self.feature=None
        self.gradient=None

    def save_gradient(self,glad):

        self.gradient=glad

    def __call__(self, x):
        image_size=(x.size(-1),x.size(-2))
        feature_maps=[]

        for i in range(x.size(0)):
            img=x[i].data.cpu().numpy()
            img=img-np.min(img)
            if np.max(img) !=0:
                img=img/np.max(img)

            feature=x[i].unsqueeze(0)

            for name ,module in self.model.named_children():
                if name=='classifier':
                    feature=feature.view(feature.size(0),-1)
                feature=module(feature)

                if name=='features':
                    feature.register_hook(self.save_gradient)
                    self.feature=feature

            classes = torch.sigmoid(feature)
            one_hot, _ = classes.max(dim=-1)
            self.model.zero_grad()
            one_hot.backward()

            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)

            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.data.cpu().numpy(), image_size)
            mask = mask - np.min(mask)

            if np.max(mask) != 0:
                mask = mask / np.max(mask)

            feature_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = feature_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
            cam = cam - np.min(cam)

            if np.max(cam) != 0:
                cam = cam / np.max(cam)

            feature_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))

        feature_maps = torch.stack(feature_maps)

        return feature_maps