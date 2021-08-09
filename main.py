#################################################### IMPORT ####################################################
import os
import cv2
import numpy as np
from PIL import Image, ImageFile
from torchvision.models import resnet50, vgg16
from pascalvoc.readxml import annotations_from_xml
from pytorch_grad_cam.utils.image import show_cam_on_image
from gradcam import image_from_path, heatmap_from_image, hit_or_miss

################################################ POINTING GAME #################################################
def pointing_game(pascalvoc_path):
    path_to_images = pascalvoc_path + 'test_images/'; list_of_images = sorted(os.listdir(path_to_images))
    path_to_labels = pascalvoc_path + 'test_labels/'; list_of_labels = sorted(os.listdir(path_to_labels))
    assert len(list_of_images) == len(list_of_labels)
    print("Number of examples: {}".format(len(list_of_labels)))
    ######################################### ITERATE OVER IMAGES ##############################################
    hits = 0; misses = 0
    # for i in range(len(list_of_images)):
    for i in range(10):
        xml_path = path_to_labels + list_of_labels[i]
        boxes, names = annotations_from_xml(xml_path)
        assert len(boxes) == len(names)
        for i in range(len(boxes)):
            box = boxes[i]; name = names[i]
            image_path = path_to_images + list_of_images[24]
            image, image_tensor = image_from_path(image_path)
            image_array = np.asarray(image)/255
            model = resnet50(pretrained=True); layer = model.layer4[-1]
            grayscale_cam, [x, y] = heatmap_from_image(image_tensor, model, layer, None)        #### TODO Category is None 
            is_hit = hit_or_miss(x, y, box, 15)
            if is_hit: hits += 1
            else: misses += 1
    print("Hits: {}".format(hits)); print("Misses: {}".format(misses)) 
    accuracy = hits/(hits + misses)*100
    return accuracy

path = '/Users/apple/Downloads/VisualNav/cvmlp-p4/pascalvoc/'
pointing_acc = pointing_game(path); print("Accuracy: {}".format(pointing_acc))