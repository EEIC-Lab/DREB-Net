from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np


def evaluate_coco(gt_file, result_file):
    # 加载真实的标注数据
    coco_gt = coco.COCO(gt_file)
    # 加载检测结果
    coco_dt = coco_gt.loadRes(result_file)
    # 创建评估对象
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.catIds = [cat['id'] for cat in coco_gt.loadCats(coco_gt.getCatIds())]
    print("Evaluating these category IDs: ", coco_eval.params.catIds)
    # 运行评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    # 打印每个类别的AP值
    coco_eval.summarize()
    
    # 打印每个类别的AP值
    print("Per-category AP: ")
    for cat_id in coco_eval.params.catIds:
        cat_id = int(cat_id)
        cat_info = coco_gt.loadCats(cat_id)
        if not cat_info:
            print(f"Warning: Category ID {cat_id} not found in the annotations.")
            continue
        category_name = cat_info[0]['name']

        # 找到当前类别在评估参数中的索引位置
        cat_index = coco_eval.params.catIds.index(cat_id)
        
        # 计算该类别的AP（IoU=0.5:0.95, area=all, maxDets=100）
        ap = coco_eval.eval['precision'][0, :, cat_index, 0, 2]
        mean_ap = np.mean(ap[ap > -1])  # 排除-1值，计算所有IoU阈值的平均AP
        # 计算该类别的AR（IoU=0.5:0.95, area=all, maxDets=100）
        ar = coco_eval.eval['recall'][0, cat_index, 2]
        mean_ar = np.mean(ar[ar > -1])  # Exclude -1 values
        # mean_ar = 0

        print(f"{category_name} (ID: {cat_id}): AP = {mean_ap:.3f}, AR = {mean_ar:.3f}")


    # 设置特定的评估条件
    coco_eval.params.iouThrs = [0.50]  # 单一IoU阈值为0.50
    # 重新进行评估和累积，专注于设置的特定条件
    coco_eval.evaluate()
    coco_eval.accumulate()

    # 显示特定条件下的结果
    print("Custom AR @[ IoU=0.50 | area=all | maxDets=100 ]: ")
    coco_eval.summarize()


# VISDrone
gt_file = 'path_to_annotation.json'  # 真实的COCO格式标注文件路径
result_file = 'path_to_results.json'

evaluate_coco(gt_file, result_file)

