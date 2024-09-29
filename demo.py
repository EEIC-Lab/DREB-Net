from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths

import os
import cv2
import numpy as np

from lib.opts import opts
from lib.detectors.ctdet_detector import CtdetDetector as Detector
from lib.datasets.dataset_factory import get_dataset
from lib.datasets.dataset.visdrone2019DET import VISDRONE_class_name as visdrone_class_name
from lib.datasets.dataset.uavdt import UAVDT_class_name as uavdt_class_name
from lib.utils.debugger import color_list

from tools.get_file_list import get_file_list

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def add_coco_bbox(show_image, bbox, cat, conf=1, show_txt=True, img_id='default', show_conf=False): 
    bbox = np.array(bbox, dtype=np.int32)
    cat = int(cat)

    colors = [(color_list[_]).astype(np.uint8) for _ in range(len(color_list))]
    colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)

    if opt.dataset == 'visdrone':
        names = visdrone_class_name
        c = colors[cat][0][0].tolist()
    elif opt.dataset == 'uavdt':
        names = uavdt_class_name
        c = colors[cat+1][0][0].tolist()

    if show_conf:
        txt = '{}{:.1f}'.format(names[cat], conf)
    else:
        txt = '{}'.format(names[cat])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(show_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
    if show_txt:
        cv2.rectangle(show_image, (bbox[0], bbox[1] - cat_size[1] - 2),
                                  (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
        cv2.putText(show_image, txt, (bbox[0], bbox[1] - 2), 
                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    return show_image



def restore_image(padded_image, original_shape):

    original_height, original_width = original_shape[:2]
    aspect_ratio = original_width / original_height
    
    if original_width > original_height:
        new_width = 1024
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 1024
        new_width = int(new_height * aspect_ratio)
    
    if new_width < 1024:
        padding_width = (1024 - new_width) // 2
        cropped_image = padded_image[:, padding_width:padding_width + new_width]
    else:
        padding_height = (1024 - new_height) // 2
        cropped_image = padded_image[padding_height:padding_height + new_height, :]

    restored_image = cv2.resize(cropped_image, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    
    return restored_image


def demo(opt):
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    detector = Detector(opt)

    if opt.demo == 'webcam':
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow('sharp_input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
                print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
            
    else:
        if os.path.isdir(opt.demo):
            image_names = get_file_list(opt.demo)

        else:
            image_names = [opt.demo]
        
        for image_name in image_names:
            print(image_name)
            image = cv2.imread(image_name)
            image_shape = image.shape
            show_image = image.copy()

            ret = detector.run(image, demo_with_deblur=opt.demo_with_deblur)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)

            results = ret['results']

            if opt.demo_with_deblur:
                show_image = ret['deblur_out']
                show_image = show_image.cpu()
                show_image = show_image.squeeze(0).numpy()
                show_image = show_image.transpose(1, 2, 0)
                show_image = ((show_image * opt.std + opt.mean) * 255.).astype(np.uint8)
                print(np.min(show_image), np.max(show_image))
                show_image = np.clip(show_image, 0, 255).astype(np.uint8)
                show_image = restore_image(show_image, image_shape)

            for j in range(1, opt.num_classes + 1):
                for bbox in results[j]:
                    if bbox[4] > opt.vis_thresh:
                        result_image = add_coco_bbox(show_image, bbox[:4], j - 1, bbox[4])

            save_name = os.path.join(opt.demo_save_path, image_name.split('/')[-1])
            print('save_name:', save_name)
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            cv2.imwrite(save_name, result_image)


if __name__ == '__main__':
    opt = opts().parse()
    demo(opt)
