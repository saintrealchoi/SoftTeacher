import os
import json
import cv2
from ssod.models.utils import Transform2D
import numpy as np
from numpy.linalg import inv
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def bbox2points(box):
    min_x, min_y, max_x, max_y = torch.split(box[:, :4], [1, 1, 1, 1], dim=1)

    return torch.cat(
        [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y], dim=1
    ).reshape(
        -1, 2
    )  # n*4,2


def points2bbox(point, max_w, max_h):
    point = point.reshape(-1, 4, 2)
    if point.size()[0] > 0:
        min_xy = point.min(dim=1)[0]
        max_xy = point.max(dim=1)[0]
        xmin = min_xy[:, 0].clamp(min=0, max=max_w)
        ymin = min_xy[:, 1].clamp(min=0, max=max_h)
        xmax = max_xy[:, 0].clamp(min=0, max=max_w)
        ymax = max_xy[:, 1].clamp(min=0, max=max_h)
        min_xy = torch.stack([xmin, ymin], dim=1)
        max_xy = torch.stack([xmax, ymax], dim=1)
        return torch.cat([min_xy, max_xy], dim=1)  # n,4
    else:
        return point.new_zeros(0, 4)

ttl_bbox = []
GT_Bbox = []

def return_pseudo_bbox(dir_name,train,i):
    img = cv2.imread(os.path.join('data','VisDrone','VisDrone2019-DET-train',train['images'][i]['file_name']))
    # img= cv2.flip(img,1)

    with open(os.path.join('work_dirs',dir_name,train['images'][i]['file_name'].split('/')[-1][:-4]+'.json'),'r') as ann:
        json_file = json.load(ann)

    gt_ann = open(os.path.join('data','VisDrone','VisDrone2019-DET-train','annotations',train['images'][i]['file_name'].split('/')[-1][:-4]+'.txt'),'r')
    gt = gt_ann.readlines()
    gt_ann.close()

    bboxes = json_file['ann']
    transform_matrix = json_file['tm']
    rev = np.array(transform_matrix)
    rev = inv(rev)
    rev = torch.from_numpy(rev)

    ttl_bbox.append(len(bboxes))
    # print("{} have {} bboxes".format(train['images'][i]['file_name'].split('/')[-1],len(bboxes)))
    for bbox in bboxes:
        # print(bbox)
        bbox = np.array(bbox)
        bbox = torch.from_numpy(bbox[:4])
        bbox = bbox.unsqueeze(0)
        points = bbox2points(bbox)
        points = torch.cat(
            [points, points.new_ones(points.shape[0], 1)], dim=1
        )  # n,3
        points = torch.matmul(rev, points.t()).t()
        points = points[:, :2] / points[:, 2:3]
        bbox = points2bbox(points, 1600, 1600)
        bbox = bbox.cpu().detach().numpy()
        bbox = bbox[0]

        cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
        cv2.putText(img,'Pseudo',(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    GT_Bbox.append(len(gt))
    # print("{} have {} GT bboxes".format(train['images'][i]['file_name'].split('/')[-1],len(gt)))

    for line in gt:
        bbox = line.split(',')
        x,y,w,h = bbox[:4]
        x1,y1,x2,y2 = int(x),int(y),int(x)+int(w),int(y)+int(h)
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
        # cv2.putText(img,'GT',(x1,y2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
        
    ann.close()
    return img,len(bboxes)
    
def show_pseudo_bbox(train):
    dir_name = ['pseudo_label','cls','reg']
    rows = 1
    columns = 3
    for i in range(len(train['images'])):
        if os.path.isfile(os.path.join('work_dirs',dir_name[0],train['images'][i]['file_name'].split('/')[-1][:-4]+'.json')):
            fig = plt.figure(figsize=(63,21))
            for j,dir in enumerate(dir_name):
                fig.add_subplot(rows,columns,j+1)

                img,num = return_pseudo_bbox(dir,train,i)
                plt.imshow(img)
                plt.axis('off')
                plt.title(dir_name[j]+": {}".format(num))
            fig.tight_layout()

            plt.show()

def main():
    with open('data/VisDrone/annotations/train.json','r') as filename:
        train = json.load(filename)
    show_pseudo_bbox(train)
    print("Average PseudoBBox Num : {}".format(len(ttl_bbox)/3))
    print("Average GT Num : {}".format(len(GT_Bbox)/3))

if __name__ == '__main__':
    main()