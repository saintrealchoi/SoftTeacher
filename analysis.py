import os
import json
import cv2
from matplotlib.font_manager import json_dump
from ssod.models.utils import Transform2D
import numpy as np
from numpy.linalg import inv
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

ious = [ list() for i in range(3) ]
    
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

    with open(os.path.join('work_dirs','pseudo_label',train['images'][i]['file_name'].split('/')[-1][:-4]+'.json'),'r') as ann:
        json_file_test = json.load(ann)

    with open(os.path.join('work_dirs',dir_name,train['images'][i]['file_name'].split('/')[-1][:-4]+'.json'),'r') as ann:
        json_file = json.load(ann)

    gt_ann = open(os.path.join('data','VisDrone','VisDrone2019-DET-train','annotations',train['images'][i]['file_name'].split('/')[-1][:-4]+'.txt'),'r')
    gt = gt_ann.readlines()
    gt_ann.close()

    bboxes = json_file['ann']
    
    transform_matrix = json_file['tm']
    test = json_file_test['tm']
    test = np.array(test)
    test = inv(test)
    test = torch.from_numpy(test)
    rev = np.array(transform_matrix)
    rev = inv(rev)
    rev = torch.from_numpy(rev)

    ttl_bbox.append(len(bboxes))
    # print("{} have {} bboxes".format(train['images'][i]['file_name'].split('/')[-1],len(bboxes)))
    for_show = []
    for bbox in bboxes:
        # print(bbox)
        bbox = np.array(bbox)
        bbox = torch.from_numpy(bbox[:4])
        bbox = bbox.unsqueeze(0)
        points = bbox2points(bbox)
        points = torch.cat(
            [points, points.new_ones(points.shape[0], 1)], dim=1
        )  # n,3
        if dir_name != "pseudo_label":
            points = torch.matmul(test, points.t()).t()
        else:
            points = torch.matmul(rev, points.t()).t()
        points = points[:, :2] / points[:, 2:3]
        bbox = points2bbox(points, 1600, 1600)
        bbox = bbox.cpu().detach().numpy()
        bbox = bbox[0]
        for_show.append(bbox)

        cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
        cv2.putText(img,'Pseudo',(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    GT_Bbox.append(len(gt))
    # print("{} have {} GT bboxes".format(train['images'][i]['file_name'].split('/')[-1],len(gt)))
    GT_show = []
    for line in gt:
        bbox = line.split(',')
        x,y,w,h = bbox[:4]
        x1,y1,x2,y2 = int(x),int(y),int(x)+int(w),int(y)+int(h)
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
        # cv2.putText(img,'GT',(x1,y2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
        GT_show.append([x1,y1,x2,y2])
    ann.close()
    if len(bboxes) == 0:
        return img,0,for_show,GT_show
    return img,len(bboxes),for_show,GT_show
    
def show_pseudo_bbox(train):
    dir_name = ['pseudo_label','cls','reg']
    rows = 1
    columns = 3
    for i in range(len(train['images'])):
        if os.path.isfile(os.path.join('work_dirs',dir_name[0],train['images'][i]['file_name'].split('/')[-1][:-4]+'.json')):
            # fig = plt.figure(figsize=(63,21))
            for j,dir in enumerate(dir_name):
                # fig.add_subplot(rows,columns,j+1)

                img,num,bboxes,GT_show = return_pseudo_bbox(dir,train,i)
                if num == 0:
                    GT_show = np.array(GT_show)
                    ious[j].append(0)
                    # plt.imshow(img)
                    # plt.axis('off')
                    # plt.title(dir_name[j]+": {}".format(num))
                    print("There is no bbox")
                    # print("{}, {} : {}".format(train['images'][i]['file_name'],j,0))    
                    continue
                bboxes = np.array(bboxes)
                bboxes = bboxes[:,:4]
                
                GT_show = np.array(GT_show)
                value = bbox_overlaps(bboxes,GT_show)
                maximum_value = np.max(value,axis=0)
                idx = np.nonzero(maximum_value)
                maximum_value = maximum_value[idx]

                if len(maximum_value) == 0:
                    # print("{}, {} : {}".format(train['images'][i]['file_name'],j,0))    
                    ious[j].append(0)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(dir_name[j]+": {}".format(num))
                    print("There is no BBox calculated")
                    continue

                # print("{}, {} : {}".format(train['images'][i]['file_name'],j,np.average(maximum_value)))    
                ious[j].append(np.average(maximum_value))
                # plt.imshow(img)
                # plt.axis('off')
                # plt.title(dir_name[j]+": {}".format(num))

            # fig.tight_layout()
            # plt.show()
            # plt.close()

        # if i == 10:
        #     break

def main():
    with open('data/VisDrone/annotations/train.json','r') as filename:
        train = json.load(filename)
    show_pseudo_bbox(train)
    # print("Average PseudoBBox Num : {}".format(sum(ttl_bbox)/3))
    # print("Average GT Num : {}".format(sum(GT_Bbox)/3))
    # print("Average ious")
    # print(ious[0])
    # print(ious[1])
    # print(ious[2])
    d = {}
    ious[1] = [float(data) for data in ious[1]]
    ious[2] = [float(data) for data in ious[2]]
    d['cls'] = list(sorted(ious[1]))
    d['reg'] = list(sorted(ious[2]))
    with open('test2.json','w') as filename:
        json.dump(d,filename)    
    print(sorted(ious[0]))
    print(sorted(ious[1]))
    print(sorted(ious[2]))
    print(np.average(np.array(ious[0])))
    print(np.average(np.array(ious[1])))
    print(np.average(np.array(ious[2])))
if __name__ == '__main__':
    main()