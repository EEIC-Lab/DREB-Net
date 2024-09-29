import os
import sys
import cv2
import numpy as np
import pycocotools.coco as coco
import numpy as np
from tqdm import tqdm

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path))
sys.path.append(os.path.join(current_path, '..'))
print(sys.path)

from lib.utils.debugger import color_list
colors = [(color_list[_]).astype(np.uint8) for _ in range(len(color_list))]
colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)

# cls: 4
UAVDT_num_classes = 3
UAVDT_class_name = ['car', 'truck', 'bus']
UAVDT_valid_ids = [1, 2, 3]
annot_path = ''
IMG_folder = ''


def _coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
    return bbox


def IMGgetitem(index):
    class_name = UAVDT_class_name
    self_coco = coco.COCO(annot_path)
    images = self_coco.getImgIds()
    num_samples = len(images)

    for index in tqdm(range(num_samples)):
        img_id = images[index]
        file_name = self_coco.loadImgs(ids=[img_id])[0]['file_name']
        ann_ids = self_coco.getAnnIds(imgIds=[img_id])
        img_path = os.path.join(IMG_folder, file_name)
        anns = self_coco.loadAnns(ids=ann_ids)
        
        img = cv2.imread(img_path)
        for ann in anns:
            bbox = _coco_box_to_bbox(ann['bbox'])
            cls_id = int(ann['category_id'])
            label = class_name[cls_id-1]
            c = colors[cls_id][0][0].tolist()

            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), c, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cat_size = cv2.getTextSize(label, font, 0.5, 2)[0]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]) - cat_size[1] - 2),(int(bbox[0]) + cat_size[0], int(bbox[1]) - 2), c, -1)
            cv2.putText(img, label, (int(bbox[0]), int(bbox[1] - 2)), font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        
        # 保存图片
        save_image(img, file_name)


def save_image(img, file_name):
    output_dir = ''  # 指定图片保存的文件夹路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, file_name)
    cv2.imwrite(save_path, img)  # 使用OpenCV保存图像


if __name__ == '__main__':
    IMGgetitem(0)
