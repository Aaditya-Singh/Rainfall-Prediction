#################################################### IMPORT ####################################################
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.models import resnet50, vgg16
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

################################################## LOAD IMAGE ##################################################
def image_from_path(path):
    img = Image.open(path).convert('RGB')
    tensor = transforms.ToTensor()(img)
    tensor = tensor.unsqueeze(0)
    return img, tensor

image_path = '/Users/apple/Downloads/VisualNav/cvmlp-p4/demo_images/dog.jpg'
image, image_tensor = image_from_path(image_path); # image.show()
print("Image shape: {}".format(image_tensor.shape))
image_array = np.asarray(image)/255

################################################ OBTAIN HEATMAP ################################################
model = resnet50(pretrained=True)
layer = model.layer4[-1]

def image_to_heatmap(tensor, model, layer, category=None):
    cam = GradCAM(model=model, target_layer=layer, use_cuda=False)
    grayscale_cam = cam(input_tensor=image_tensor, target_category=category)
    grayscale_cam = grayscale_cam[0, :]
    max_indices = np.where(grayscale_cam == grayscale_cam.max())
    max_x = max_indices[0][0]; max_y = max_indices[1][0]
    return grayscale_cam, [max_x, max_y]

grayscale_cam, [x, y] = image_to_heatmap(image_tensor, model, layer, 230)
heatmap_array = show_cam_on_image(image_array, grayscale_cam, use_rgb=True)
heatmap_image = Image.fromarray(heatmap_array); heatmap_image.show()
print("Maximum intensity point: {}, {}".format(x, y))

################################################ POINTING GAME #################################################
def hit_or_miss(x_map, y_map, bnd_box, threshold=0):
    x_min = bnd_box['xmin']; y_min = bnd_box['ymin']; x_max = bnd_box['xmax']; y_max = bnd_box['ymax']
    if (x_map < x_min - threshold) or (x_map > x_max + threshold): return False 
    elif (y_map < y_min - threshold) or (y_map > y_max + threshold): return False
    return True

box = {'xmin': 38, 'ymin': 19, 'xmax': 385, 'ymax': 373}
is_hit = hit_or_miss(x, y, box, 15); print("Hit: {}".format(is_hit))
