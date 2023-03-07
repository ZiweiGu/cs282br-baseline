import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from helpers import *


def load_data(url):
    """
    Load and transform input images at the given url (folder)
    """
    # The specific numbers used in normalization come from the mean and std of ImageNet.
    # Here, we are making the hypothesis that our input images are similar to ImageNet images.
    return torch.utils.data.DataLoader(
    datasets.ImageFolder(url, transform=transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])])),
    batch_size= 5, shuffle=False)


def get_gradients(model, img):
    """
    Compute input gradients for an image
    """
    # Apply model to image
    out = model(img)
    # Find the predicted class for each input in a batch
    predicted_label = out.argmax(dim=1)
    # Use the negative log likelihood loss
    loss_function = F.nll_loss
    loss = -1. * loss_function(out, predicted_label, reduction='sum')
    # Explicitly set the gradients to zero so that we do the parameter update correctly
    model.zero_grad()
    # Compute gradients
    return torch.autograd.grad(outputs=loss, inputs=img, only_inputs=True)[0]


def get_saliency(model, img):
    """
    Implement the SmoothGrad saliency algorithm to compute smoothed saliency values
    Input is tensor of size (3,H,W); Output is tensor of size (1,H,W)
    """
    # Set the model to evaluation mode during inference
    model.eval()
    # Set std spread factor to be 0.15
    std_dev = 0.15 * (img.max().item() - img.min().item())
    # Initialize the class activation map (CAM) that highlights regions contributing the most to the predicted class
    cam = torch.zeros_like(img).to(img.device)
    # Add Gaussian noise to image multiple times
    for _ in range(100):
        noise = torch.normal(mean=torch.zeros_like(img).to(img.device), std=std_dev)
        cam += (get_gradients(model, img + noise)) / 100
    # Compute sum of the absolute values of activation across spatial locations 
    # to ensure both positive and negative activations contribute to the normalization
    return cam.abs().sum(1, keepdim=True)
                

if __name__ == "__main__":
    # Loading data and saving outputs, code reused from https://github.com/idiap/fullgrad-saliency
    root = os.path.dirname(os.path.abspath('baseline.py')) + '/'
    input_path = root + 'dataset/'
    output_path = root + 'results/'
    create_folder(output_path)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    # Use the pre-trained resnet model with the most up-to-date weights
    model = models.resnet18(weights='IMAGENET1K_V1').to(device)
    # Undo normalization on images. Again, the numbers come from the mean and std of ImageNet.
    unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225])
    for batch_idx, (data, _) in enumerate(load_data(input_path)):
        # Track the operations performed on the tensor and compute the gradients automatically during the backward pass.
        data = data.to(device).requires_grad_()
        # Compute saliency maps for the input image
        saliency_map = get_saliency(model, data)
        # Save saliency maps
        for i in range(data.size(0)):
            filename = output_path + str((batch_idx+1) * (i+1))
            image = unnormalize(data[i].cpu())
            save_saliency_map(image, saliency_map[i], filename + '_' + '.jpg') # Add complete path and file extension
    print('Finished generating saliency maps.')
