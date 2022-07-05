import json
import os


with open('/home/choisj/git/test/SoftTeacher/data/VisDrone/annotations/semi_supervised/instances_train2021.1@10.json','r') as json_file:
    f = json.load(json_file)

with open('/home/choisj/git/test/SoftTeacher/data/VisDrone/annotations/val.json','r') as json_file:
    f2 = json.load(json_file)
with open('/home/choisj/git/test/SoftTeacher/data/VisDrone/annotations/train.json','r') as json_file:
    f3 = json.load(json_file)
print(len(f['images']))
print(len(f2['images']))
print(len(f3['images']))
print(len(f3['images'])-len(f['images']))