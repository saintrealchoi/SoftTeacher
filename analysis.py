import pickle
from podm.metrics import MetricPerClass

ttl = 0

# for i in range(10):
#     with open('/home/choisj/Desktop/analysis_256_'+str(i)+'.json', 'rb') as fp:
#         results = pickle.load(fp)

#     mAP = MetricPerClass.mAP(results)
#     ttl += mAP
#     print(mAP)

# print()
# print(ttl/10)

# ap = []
# tp = []
# fp = []

# print(label['airplane'])


with open('/home/choisj/Desktop/analysis.json', 'rb') as fp:
    label = pickle.load(fp)
ttl=0
for i in range(10):
    mAP= 0
    tmp_ap = 0
    tmp_tp = 0
    tmp_fp = 0
    num_gt = 0
    for j,cls in label.items():
        tmp_tp += cls['tp'][i]
        tmp_fp += cls['fp'][i]
        mAP += cls['ap'][i]
        num_gt += cls['num_groundtruth'][i]
        # print(cls['ap'][i])
    print("{}th tp= {}".format(i,tmp_tp))
    print("{}th fp= {}".format(i,tmp_fp))
    print("{}th gt= {}".format(i,num_gt))
    print("mAP = {}".format(mAP/80))
    ttl+=mAP/80
print(ttl/10)