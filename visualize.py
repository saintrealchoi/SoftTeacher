import cv2

img = cv2.imread('/home/choisj/Downloads/VisDrone/VisDrone2019-DET-train/images/0000013_00465_d_0000067.jpg',cv2.IMREAD_COLOR)
f = open("/home/choisj/Downloads/VisDrone/VisDrone2019-DET-train/annotations/0000013_00465_d_0000067.txt", 'r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    # print(line)
    real = line.split(',')
    print(real)
    cv2.rectangle(img,(int(real[0]),int(real[1])),(int(real[0])+int(real[2]),int(real[1])+int(real[3])),(0,0,255),1)
print(img.shape)
cv2.imshow('image', img)
cv2.imwrite("/home/choisj/Desktop/visdrone2.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
f.close()
