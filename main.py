#################################################### IMPORT ####################################################
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.models import resnet50, vgg16
from readxml import annotations_from_xml
from pytorch_grad_cam.utils.image import show_cam_on_image
from gradcam import image_from_path, heatmap_from_image, hit_or_miss

################################################ POINTING GAME #################################################
def pointing_game(dataset_path):
    path_to_images = dataset_path + 'images/'; list_of_images = sorted(os.listdir(path_to_images))
    path_to_labels = dataset_path + 'labels/'; list_of_labels = sorted(os.listdir(path_to_labels))
    assert len(list_of_images) == len(list_of_labels)
    print("Number of examples: {}".format(len(list_of_labels)))
    df = pd.read_csv('/Users/apple/Downloads/VisualNav/cvmlp-p4/imagenet/synset_to_id.txt', sep = ":", header = None)
    category_dict = {}
    for i in range(len(df)): category_dict[df.iloc[i, 0]] = df.iloc[i, 1]
    ######################################### ITERATE OVER IMAGES ##############################################
    hits = 0; misses = 0
    # for i in tqdm(range(len(list_of_images))):
    for i in tqdm(range(100)):
        image_path = path_to_images + list_of_images[i]
        image, image_tensor = image_from_path(image_path); image_array = np.asarray(image)/255
        xml_path = path_to_labels + list_of_labels[i]
        boxes, names = annotations_from_xml(xml_path)
        assert len(boxes) == len(names)
        for j in range(len(boxes)):
            box = boxes[j]; name = names[j]
            category = int(category_dict[name])
            model = resnet50(pretrained=True); layer = model.layer4[-1]
            grayscale_cam, [x, y] = heatmap_from_image(image_tensor, model, layer, category)
            # heatmap_array = show_cam_on_image(image_array, grayscale_cam, use_rgb=True)
            # heatmap_image = Image.fromarray(heatmap_array); heatmap_image.show() 
            is_hit = hit_or_miss(x, y, box, 15)
            if is_hit: hits += 1
            else: misses += 1
    print("Hits: {}".format(hits)); print("Misses: {}".format(misses)) 
    accuracy = hits/(hits + misses)*100
    return accuracy

# path = '/Users/apple/Downloads/VisualNav/cvmlp-p4/pascalvoc/'
path = '/Users/apple/Downloads/VisualNav/cvmlp-p4/imagenet/'
pointing_acc = pointing_game(path); print("Accuracy: {}".format(pointing_acc))