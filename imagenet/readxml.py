import xmltodict
import pandas as pd
import cv2

path = open('/Users/apple/Downloads/VisualNav/cvmlp-p4/imagenet/ILSVRC2012_val_00000003.xml', 'r')
xml = path.read(); # print(xml)
dictionary = xmltodict.parse(xml)
synset = dictionary['annotation']['object']['name']; box = {}
box['xmin'] = int(dictionary['annotation']['object']['bndbox']['xmin'])
box['ymin'] = int(dictionary['annotation']['object']['bndbox']['ymin'])
box['xmax'] = int(dictionary['annotation']['object']['bndbox']['xmax'])
box['ymax'] = int(dictionary['annotation']['object']['bndbox']['ymax'])
print(synset); print(box)

df = pd.read_csv('/Users/apple/Downloads/VisualNav/cvmlp-p4/imagenet/synset_to_id.txt', sep=':', header=None)
synset_to_id = {}
for i in range(len(df)): synset_to_id[df.iloc[i, 0]] = df.iloc[i, 1]
ids = synset_to_id[synset]
print(ids)

image = cv2.imread('/Users/apple/Downloads/VisualNav/cvmlp-p4/demo_images/ILSVRC2012_val_00000003.JPEG')
cv2.rectangle(image, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), color=(0, 255, 0), thickness=2)
cv2.imshow("bounded", image); cv2.waitKey(0)