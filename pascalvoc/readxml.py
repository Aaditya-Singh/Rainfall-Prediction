import xmltodict
import pandas as pd
import cv2

########################################## BNDBOXES FOR OBJECTS IN XML #########################################
def annotations_from_xml(xml_path):
    xml = path.read(); # print(xml)
    dicts = xmltodict.parse(xml)
    list_of_boxes = []; list_of_names = []
    for i in range(len(dicts['annotation']['object'])):
        object_dict = dicts['annotation']['object'][i]
        name = object_dict['name']; box = {}
        box['xmin'] = int(object_dict['bndbox']['xmin'])
        box['ymin'] = int(object_dict['bndbox']['ymin'])
        box['xmax'] = int(object_dict['bndbox']['xmax'])
        box['ymax'] = int(object_dict['bndbox']['ymax'])
        list_of_boxes.append(box); list_of_names.append(name)
        print("Class: {}".format(name)); print("Box: {}".format(box))
        #################################### show image with bndbox ####################################
        image = cv2.imread('/Users/apple/Downloads/VisualNav/cvmlp-p4/demo_images/human.jpg')
        cv2.rectangle(image, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), color=(0, 255, 0), thickness=2)
        cv2.imshow("bounded", image); cv2.waitKey(0)
    return list_of_boxes, list_of_names

path = open('/Users/apple/Downloads/VisualNav/cvmlp-p4/pascalvoc/human.xml', 'r')
boxes, names = annotations_from_xml(path)
print("Clases: {}".format(names)); print("Boxes: {}".format(boxes))