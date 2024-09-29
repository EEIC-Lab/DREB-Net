# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random

# 水平翻转图像
def flip(img):
    return img[:, :, ::-1].copy()  


# 仿射变换结果，转换到一个新的坐标系统中
def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


# 仿射变换
def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]	#基于缩放因子的源宽度
    dst_w = output_size[0]	#目标输出的宽度
    dst_h = output_size[1]	#目标输出的高度

    rot_rad = np.pi * rot / 180		#将旋转角度从度转换为弧度
    src_dir = get_dir([0, src_w * -0.5], rot_rad)	#计算源方向向量，考虑旋转和缩放因子。
    dst_dir = np.array([0, dst_w * -0.5], np.float32)	#计算目标方向向量，只考虑目标尺寸。

    # src和dst数组分别存储变换前后的坐标点。
    # 第一个点是中心点经过平移后的位置。
    # 第二个点是通过中心点加上方向向量（考虑旋转和缩放）得到的点。
    # 第三个点是通过get_3rd_point函数计算出来，这个函数基于前两个点计算第三个点以保持仿射变换的定义。
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


# 对单个点pt应用仿射变换矩阵t
def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


# 给定两个点a和b，计算第三个点，使得这三个点可以构成一个仿射变换需要的三点系统
def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


#根据旋转角度rot_rad，计算一个点src_point绕原点旋转后的新位置
def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


# 对图像进行裁剪和变换，包括缩放和旋转
def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


# 计算高斯核的半径，使得生成的高斯分布覆盖的区域能满足特定的重叠率要求
# 这在目标检测中创建理想的热图时非常有用，以确保目标区域被适当地标记
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


# 生成二维高斯分布核，通常用于图像处理中的模糊操作或生成热图
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


# 在热图heatmap上，以center为中心，使用指定的radius和权重k绘制高斯核
# 这种方法在目标检测算法中用于强调目标的中心位置
def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
        
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


# 这个函数在regmap上，根据heatmap的热点位置center，以高斯分布的方式调整区域内的预测值。用于细化目标检测中的位置预测。
# 可能存在错误，diameter*2+1应为diameter+1 !!!!!!!!!!!但也可能是为了处理更大范围内的偏移
def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter*2+1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)
    
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                                radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                        radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
        1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


# 根据MSRA方法在热图heatmap上绘制高斯核。这通常用于生成更精确的热图标记，其中sigma控制高斯核的扩散程度
def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(aug_params, image, eigval, eigvec):
    alpha = aug_params['lighting']
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(image, aug_params, gs, gs_mean):
    alpha = aug_params['saturation']
    blend_(alpha, image, gs[:, :, None])


def brightness_(image, aug_params, gs, gs_mean):
    alpha = aug_params['brightness']
    image *= alpha


def contrast_(image, aug_params, gs, gs_mean):
    alpha = aug_params['contrast']
    blend_(alpha, image, gs_mean)


def color_aug(data_rng, image1, image2, eig_val, eig_vec):
    aug_params = generate_aug_params(data_rng)
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)
    
    gs1 = grayscale(image1)
    gs_mean = gs1.mean()
    for f in functions:
        f(image1, aug_params, gs1, gs_mean)
    lighting_(aug_params, image1, eig_val, eig_vec)

    gs2 = grayscale(image2)
    gs_mean = gs2.mean()
    for f in functions:
        f(image2, aug_params, gs2, gs_mean)
    lighting_(aug_params, image2, eig_val, eig_vec)


def generate_aug_params(data_rng):
    params = {
        'brightness': 1. + data_rng.uniform(low=-0.4, high=0.4),
        'contrast': 1. + data_rng.uniform(low=-0.4, high=0.4),
        'saturation': 1. + data_rng.uniform(low=-0.4, high=0.4),
        'lighting': data_rng.normal(scale=0.1, size=(3,))
    }
    return params

