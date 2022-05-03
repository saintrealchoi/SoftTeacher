import json

with open('/home/choisj/Downloads/dataset/coco/annotations/instances_val2017.json','r') as file:
    ann = json.load(file)
print(ann)
