import xmltodict
import pandas as pd
import cv2

path = open('/Users/apple/Downloads/VisualNav/cvmlp-p4/pascalvoc/cat.xml', 'r')
xml = path.read(); # print(xml)
dictionary = xmltodict.parse(xml)
name = dictionary['annotation']['object']['name']; box = {}
box['xmin'] = int(dictionary['annotation']['object']['bndbox']['xmin'])
box['ymin'] = int(dictionary['annotation']['object']['bndbox']['ymin'])
box['xmax'] = int(dictionary['annotation']['object']['bndbox']['xmax'])
box['ymax'] = int(dictionary['annotation']['object']['bndbox']['ymax'])
print(name); print(box)

image = cv2.imread('/Users/apple/Downloads/VisualNav/cvmlp-p4/demo_images/cat.jpg')
cv2.rectangle(image, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), color=(0, 255, 0), thickness=2)
cv2.imshow("bounded", image); cv2.waitKey(0)