# Visual detection result on KITTI dataset

import os
import cv2
import glob


def draw(txt_file, im):
    with open(txt_file, 'rb') as f:
        lines = f.readlines()
    for line in lines:
        lists = line.strip().split(' ')
        cls = lists[0]
        x1 = int(float(lists[4]))
        y1 = int(float(lists[5]))
        x2 = int(float(lists[6]))
        y2 = int(float(lists[7]))
        score = float(lists[-1])
        if score > 0.7:
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = str(cls) + ": " + str(format(score*100, '.2f'))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im, text, (x1, y1), font, .4, (0, 0, 255), 1)
    return im


kitti_test_img_dir = '/media/yangshun/0008EB70000B0B9F/PycharmProjects/faster_rcnn/data/demo/web_car'
kitti_test_txt_dir = '/media/yangshun/0008EB70000B0B9F/roi_roi_new/kitti/txt/debug/data'

images = glob.glob(os.path.join(kitti_test_img_dir, '*'))
# images = sorted(images)

for img in images:
    name = os.path.basename(img).split('.')[0]
    txt = os.path.join(kitti_test_txt_dir, name + '.txt')
    im = cv2.imread(img)
    im = draw(txt, im)
    cv2.imshow('0', im)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('/media/yangshun/0008EB70000B0B9F/PycharmProjects/faster_rcnn/data/demo/result/'+name+'.jpg', im)
cv2.destroyAllWindows()

