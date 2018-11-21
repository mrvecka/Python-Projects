import os
import json

annotation_file = 'D:\\dataset\\COCO\\annotations_updated\\instances_train2017.json'
image_dir = 'D:\\dataset\\COCO\\train2017_updated\\'
dataset = json.load(open(annotation_file, 'r'))
for i in range(len(dataset['images'])):
    file_name = dataset['images'][i]['file_name']
    ids = dataset['images'][i]['id']
    if not os.access(image_dir + file_name,os.R_OK):
        anno = [item for item in dataset['annotations'] if item['id']==ids]
        if len(anno) > 0:
            dataset['annotations'].remove(anno[0])
            
        dataset['images'].remove(dataset['images'][i])


file = open(annotation_file,'w+')
file.write(dataset)
file.close()
print(dataset)