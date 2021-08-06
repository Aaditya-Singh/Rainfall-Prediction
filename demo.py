#################################################### IMPORT ####################################################
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.models import resnet50, vgg16
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

################################################## LOAD IMAGE ##################################################
def path_to_image(path):
    img = Image.open(path).convert('RGB')
    tensor = transforms.ToTensor()(img)
    tensor = tensor.unsqueeze(0)
    return img, tensor

image_path = '/Users/apple/Downloads/VisualNav/pytorch-grad-cam/demo_images/both.png'
image, image_tensor = path_to_image(image_path); # image.show()
print(image_tensor.shape)
image_array = np.asarray(image)/255

################################################ DEFINE MODEL ##################################################
model = resnet50(pretrained=True)
layer = model.layer4[-1]

cam = GradCAM(model=model, target_layer=layer, use_cuda=False)
category = None
grayscale_cam = cam(input_tensor=image_tensor, target_category=category)
grayscale_cam = grayscale_cam[0, :]

heatmap_array = show_cam_on_image(image_array, grayscale_cam, use_rgb=True)
heatmap_image = Image.fromarray(heatmap_array)
heatmap_image.show()
