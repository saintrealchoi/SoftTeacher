from podm import coco_decoder
from podm.metrics import get_pascal_voc_metrics, MetricPerClass, get_bounding_boxes
import mmcv
import pickle
# from thirdparty.mmdetection.mmdet.datasets.coco import CocoDataset


# result = CocoDataset.format_results()

with open('./data/coco/annotations/instances_val2017.json') as fp:
    gold_dataset = coco_decoder.load_true_bounding_box_dataset(fp)


with open('/home/choisj/Desktop/result_512.bbox.json') as fp:
    pred_dataset = coco_decoder.load_pred_bounding_box_dataset(fp, gold_dataset)


gt_BoundingBoxes = get_bounding_boxes(gold_dataset)
pd_BoundingBoxes = get_bounding_boxes(pred_dataset)

label = {}
for i in range(10):
    thr = 0.5 + i*0.05
    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, thr)

    for cls, metric in results.items():
        label_cls = metric.label
        label.setdefault(label_cls, {})
        # label = metric.label
        # label.label_cls.append
        label[label_cls].setdefault('ap',[])
        label[label_cls].setdefault('tp',[])
        label[label_cls].setdefault('fp',[])
        label[label_cls].setdefault('num_groundtruth',[])
        label[label_cls].setdefault('num_detection',[])
        label[label_cls]['ap'].append(metric.ap)
        label[label_cls]['tp'].append(metric.tp)
        label[label_cls]['fp'].append(metric.fp)
        label[label_cls]['num_groundtruth'].append(metric.num_groundtruth)
        label[label_cls]['num_detection'].append(metric.num_detection)
        # label.labe
        # print('ap', metric.ap)
        # print('precision', metric.precision)
        # print('interpolated_recall', metric.interpolated_recall)
        # print('interpolated_precision', metric.interpolated_precision)
        # print('tp', metric.tp)
        # print('fp', metric.fp)
        # print('num_groundtruth', metric.num_groundtruth)
        # print('num_detection', metric.num_detection)

with open('/home/choisj/Desktop/analysis_512.json', 'wb') as fp:
    pickle.dump(label, fp)
print(label)



# for i in range(10):
#     results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes,0.5+0.05*i)
#     with open('/home/choisj/Desktop/analysis_256_'+str(i)+'.json', 'wb') as fp:
#         pickle.dump(results, fp)
# # print(results)

# gt = 0
# ap = 0
# for cls, metric in results.items():
#     # label = metric.label
#     ap += metric.ap
#     print('ap', metric.ap)
#     # print('precision', metric.precision)
#     # print('interpolated_recall', metric.interpolated_recall)
#     # print('interpolated_precision', metric.interpolated_precision)
#     # print('tp', metric.tp)
#     # print('fp', metric.fp)
#     # print('num_groundtruth', metric.num_groundtruth)
#     # print('num_detection', metric.num_detection)
#     gt+=metric.num_groundtruth
# print(ap)
# print(gt)



# # # for 
    
# mAP = MetricPerClass.mAP(results)
# print(mAP)



# # for 