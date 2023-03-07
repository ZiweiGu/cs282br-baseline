import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils, models

from helpers import *


class SmoothGrad():
    """
    Compute smoothgrad 
    """

    def __init__(self, model, num_samples=100, std_spread=0.15):
        self.model = model
        self.num_samples = num_samples
        self.std_spread = std_spread

    def _getGradients(self, image, target_class=None):
        """
        Compute input gradients for an image
        """

        image = image.requires_grad_()
        out = self.model(image)

        if target_class is None:
            target_class = out.data.max(1, keepdim=True)[1]
            target_class = target_class.flatten()

        loss = -1. * F.nll_loss(out, target_class, reduction='sum')

        self.model.zero_grad()
        # Gradients w.r.t. input and features
        input_gradient = torch.autograd.grad(outputs = loss, inputs = image, only_inputs=True)[0]

        return input_gradient

    def saliency(self, image, target_class=None):
        #SmoothGrad saliency
        
        self.model.eval()

        grad = self._getGradients(image, target_class=target_class)
        std_dev = self.std_spread * (image.max().item() - image.min().item())

        cam = torch.zeros_like(image).to(image.device)
        # add gaussian noise to image multiple times
        for i in range(self.num_samples):
            noise = torch.normal(mean = torch.zeros_like(image).to(image.device), std = std_dev)
            cam += (self._getGradients(image + noise, target_class=target_class)) / self.num_samples

        return cam.abs().sum(1, keepdim=True)
    


PATH = os.path.dirname(os.path.abspath('baseline.py')) + '/'
sample_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(PATH + 'dataset/', transform=transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
                   ])),
    batch_size= 5, shuffle=False)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
model = models.resnet18(pretrained=True)
model = model.to(device)

unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225])


if __name__ == "__main__":
    save_path = PATH + 'results/'
    create_folder(save_path)
    for batch_idx, (data, _) in enumerate(sample_loader):
        data = data.to(device).requires_grad_()

        # Compute saliency maps for the input data
        saliency_map = SmoothGrad(model).saliency(data)

        # Save saliency maps
        for i in range(data.size(0)):
            filename = save_path + str( (batch_idx+1) * (i+1))
            image = unnormalize(data[i].cpu())
            save_saliency_map(image, saliency_map[i], filename + '_' + '.jpg')
    print('Saliency maps saved in the results folder.')
