import mmcv
import pickle


with open('/home/choisj/Desktop/num_tp.pkl', 'rb') as f:
    outputs = pickle.load(f)

# num_tp = (T,K,A,M) = (10, 80, 4, 3)
num_tp = outputs['num_tp']
num_fp = outputs['num_fp']

print(num_tp.shape)

print(num_tp[0,0,0,0])
print(num_fp[0,0,0,0])

mAP =0
for j in range(10):
    ttl_tp_num = 0
    ttl_fp_num = 0
    ap = 0
    for i in range(80):
        ttl_tp_num += num_tp[j,i,0,2]
        ttl_fp_num += num_fp[j,i,0,2]
        ap += num_tp[j,i,0,2]/(num_tp[j,i,0,2]+num_fp[j,i,0,2])
    print("ap = {}".format(ap/80))
    mAP += (ap/80)
    print("Thr={}, TP={}".format(0.5+j*0.05,ttl_tp_num))
    print("Thr={}, FP={}".format(0.5+j*0.05,ttl_fp_num))
    print("Precision = {}".format(ttl_tp_num/(ttl_tp_num+ttl_fp_num)))
    print("==========================================")
    # mAP +=(ttl_tp_num/(ttl_tp_num+ttl_fp_num))
print("mAP:{}".format(mAP/10))