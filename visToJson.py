import os
import json

# def visToJson(dir_path):

#     for file in os.listdir(dir_path):
#         f = open(os.path.join(dir_path,file), 'r')
#         lines = f.readlines()
#         for line in lines:
            

        

with open('/home/choisj/Downloads/dataset/coco/annotations/instances_train2017.json','r') as f:
    file = json.load(f)

print(file)