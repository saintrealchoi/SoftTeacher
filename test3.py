from podm import coco_decoder
from podm.metrics import get_pascal_voc_metrics, MetricPerClass, get_bounding_boxes
import mmcv
import pickle
# from thirdparty.mmdetection.mmdet.datasets.coco import CocoDataset


# result = CocoDataset.format_results()

with open('./data/coco/annotations/instances_val2017.json') as fp:
    gold_dataset = coco_decoder.load_true_bounding_box_dataset(fp)


with open('/home/choisj/Desktop/result.bbox.json') as fp:
    pred_dataset = coco_decoder.load_pred_bounding_box_dataset(fp, gold_dataset)


gt_BoundingBoxes = get_bounding_boxes(gold_dataset)
pd_BoundingBoxes = get_bounding_boxes(pred_dataset)


results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)

for cls, metric in results.items():
    print('ap', metric.ap)
    # print('precision', metric.precision)
    # print('interpolated_recall', metric.interpolated_recall)
    # print('interpolated_precision', metric.interpolated_precision)
    print('tp', metric.tp)
    print('fp', metric.fp)
    print('num_groundtruth', metric.num_groundtruth)
    print('num_detection', metric.num_detection)

mAP = MetricPerClass.mAP(results)