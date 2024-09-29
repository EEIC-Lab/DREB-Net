from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from lib.datasets.dataset.visdrone2019DET import VISDRONE_class_name as visdrone_class_name
from lib.datasets.dataset.uavdt import UAVDT_class_name as uavdt_class_name

class Debugger(object):
    def __init__(self, ipynb=False, theme='black', 
                             num_classes=-1, dataset=None, down_ratio=4):
        self.ipynb = ipynb
        if not self.ipynb:
            import matplotlib.pyplot as plt
            self.plt = plt
        self.imgs = {}
        self.theme = theme
        colors = [(color_list[_]).astype(np.uint8) \
                        for _ in range(len(color_list))]
        self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
        if self.theme == 'white':
            self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
            self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)
        self.dim_scale = 1

        if dataset == 'visdrone':
            self.names = visdrone_class_name
        elif dataset =='uavdt':
            self.names = uavdt_class_name
        num_classes = len(self.names)
        
        self.down_ratio=down_ratio
        # for bird view
        self.world_size = 64
        self.out_size = 384

    def add_img(self, img, img_id='default', revert_color=False):
        if revert_color:
            img = 255 - img
        self.imgs[img_id] = img.copy()
    
    def add_mask(self, mask, bg, imgId = 'default', trans = 0.8):
        self.imgs[imgId] = (mask.reshape(
            mask.shape[0], mask.shape[1], 1) * 255 * trans + \
            bg * (1 - trans)).astype(np.uint8)
    
    def show_img(self, pause = False, imgId = 'default'):
        cv2.imshow('{}'.format(imgId), self.imgs[imgId])
        if pause:
            cv2.waitKey()
    
    def add_blend_img(self, back, fore, img_id='blend', trans=0.7):
        if self.theme == 'white':
            fore = 255 - fore
        if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
            fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
        if len(fore.shape) == 2:
            fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
        self.imgs[img_id] = (back * (1. - trans) + fore * trans)
        self.imgs[img_id][self.imgs[img_id] > 255] = 255
        self.imgs[img_id][self.imgs[img_id] < 0] = 0
        self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

    '''
    # slow version
    def gen_colormap(self, img, output_res=None):
        # num_classes = len(self.colors)
        img[img < 0] = 0
        h, w = img.shape[1], img.shape[2]
        if output_res is None:
            output_res = (h * self.down_ratio, w * self.down_ratio)
        color_map = np.zeros((output_res[0], output_res[1], 3), dtype=np.uint8)
        for i in range(img.shape[0]):
            resized = cv2.resize(img[i], (output_res[1], output_res[0]))
            resized = resized.reshape(output_res[0], output_res[1], 1)
            cl = self.colors[i] if not (self.theme == 'white') \
                     else 255 - self.colors[i]
            color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
        return color_map
        '''

    
    def gen_colormap(self, img, output_res=None):
        img = img.copy()
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        if output_res is None:
            output_res = (h * self.down_ratio, w * self.down_ratio)
        img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
        colors = np.array(
            self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
        if self.theme == 'white':
            colors = 255 - colors
        color_map = (img * colors).max(axis=2).astype(np.uint8)
        color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
        return color_map
        

    def add_coco_bbox(self, bbox, cat, conf=1, show_txt=True, img_id='default'): 
        bbox = np.array(bbox, dtype=np.int32)
        # cat = (int(cat) + 1) % 80
        cat = int(cat)
        # print('cat', cat, self.names[cat])
        c = self.colors[cat][0][0].tolist()
        if self.theme == 'white':
            c = (255 - np.array(c)).tolist()
        txt = '{}{:.1f}'.format(self.names[cat], conf)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        cv2.rectangle(
            self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
        if show_txt:
            cv2.rectangle(self.imgs[img_id],
                                        (bbox[0], bbox[1] - cat_size[1] - 2),
                                        (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
            cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - 2), 
                                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)



    def show_all_imgs(self, pause=False, time=0):
        if not self.ipynb:
            for i, v in self.imgs.items():
                cv2.imshow('{}'.format(i), v)
            if cv2.waitKey(0 if pause else 1) == 27:
                import sys
                sys.exit(0)
        else:
            self.ax = None
            nImgs = len(self.imgs)
            fig=self.plt.figure(figsize=(nImgs * 10,10))
            nCols = nImgs
            nRows = nImgs // nCols
            for i, (k, v) in enumerate(self.imgs.items()):
                fig.add_subplot(1, nImgs, i + 1)
                if len(v.shape) == 3:
                    self.plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
                else:
                    self.plt.imshow(v)
            self.plt.show()

    def save_img(self, imgId='default', path='./cache/debug/'):
        cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])
        
    def save_all_imgs(self, path='./cache/debug/', prefix='', genID=False):
        if genID:
            try:
                idx = int(np.loadtxt(path + '/id.txt'))
            except:
                idx = 0
            prefix=idx
            np.savetxt(path + '/id.txt', np.ones(1) * (idx + 1), fmt='%d')
        for i, v in self.imgs.items():
            cv2.imwrite(path + '/{}{}.png'.format(prefix, i), v)

    def remove_side(self, img_id, img):
        if not (img_id in self.imgs):
            return
        ws = img.sum(axis=2).sum(axis=0)
        l = 0
        while ws[l] == 0 and l < len(ws):
            l+= 1
        r = ws.shape[0] - 1
        while ws[r] == 0 and r > 0:
            r -= 1
        hs = img.sum(axis=2).sum(axis=1)
        t = 0
        while hs[t] == 0 and t < len(hs):
            t += 1
        b = hs.shape[0] - 1
        while hs[b] == 0 and b > 0:
            b -= 1
        self.imgs[img_id] = self.imgs[img_id][t:b+1, l:r+1].copy()

    def project_3d_to_bird(self, pt):
        pt[0] += self.world_size / 2
        pt[1] = self.world_size - pt[1]
        pt = pt * self.out_size / self.world_size
        return pt.astype(np.int32)

    def add_ct_detection(
        self, img, dets, show_box=False, show_txt=True, 
        center_thresh=0.5, img_id='det'):
        # dets: max_preds x 5
        self.imgs[img_id] = img.copy()
        if type(dets) == type({}):
            for cat in dets:
                for i in range(len(dets[cat])):
                    if dets[cat][i, 2] > center_thresh:
                        cl = (self.colors[cat, 0, 0]).tolist()
                        ct = dets[cat][i, :2].astype(np.int32)
                        if show_box:
                            w, h = dets[cat][i, -2], dets[cat][i, -1]
                            x, y = dets[cat][i, 0], dets[cat][i, 1]
                            bbox = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                                                            dtype=np.float32)
                            self.add_coco_bbox(
                                bbox, cat - 1, dets[cat][i, 2], 
                                show_txt=show_txt, img_id=img_id)
        else:
            for i in range(len(dets)):
                if dets[i, 2] > center_thresh:
                    # print('dets', dets[i])
                    cat = int(dets[i, -1])
                    cl = (self.colors[cat, 0, 0] if self.theme == 'black' else \
                                                                             255 - self.colors[cat, 0, 0]).tolist()
                    ct = dets[i, :2].astype(np.int32) * self.down_ratio
                    cv2.circle(self.imgs[img_id], (ct[0], ct[1]), 3, cl, -1)
                    if show_box:
                        w, h = dets[i, -3] * self.down_ratio, dets[i, -2] * self.down_ratio
                        x, y = dets[i, 0] * self.down_ratio, dets[i, 1] * self.down_ratio
                        bbox = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                                                        dtype=np.float32)
                        self.add_coco_bbox(bbox, dets[i, -1], dets[i, 2], img_id=img_id)


    def add_2d_detection(
        self, img, dets, show_box=False, show_txt=True, 
        center_thresh=0.5, img_id='det'):
        self.imgs[img_id] = img
        for cat in dets:
            for i in range(len(dets[cat])):
                cl = (self.colors[cat - 1, 0, 0]).tolist()
                if dets[cat][i, -1] > center_thresh:
                    bbox = dets[cat][i, 1:5]
                    self.add_coco_bbox(
                        bbox, cat - 1, dets[cat][i, -1], 
                        show_txt=show_txt, img_id=img_id)



color_list = np.array(
                [
                        1.000, 1.000, 1.000,
                        0.850, 0.325, 0.098,
                        0.929, 0.694, 0.125,
                        0.494, 0.184, 0.556,
                        0.466, 0.674, 0.188,
                        0.301, 0.745, 0.933,
                        0.635, 0.078, 0.184,
                        0.300, 0.300, 0.300,
                        0.600, 0.600, 0.600,
                        1.000, 0.000, 0.000,
                        1.000, 0.500, 0.000,
                        0.749, 0.749, 0.000,
                        0.000, 1.000, 0.000,
                        0.000, 0.000, 1.000,
                        0.667, 0.000, 1.000,
                        0.333, 0.333, 0.000,
                        0.333, 0.667, 0.000,
                        0.333, 1.000, 0.000,
                        0.667, 0.333, 0.000,
                        0.667, 0.667, 0.000,
                        0.667, 1.000, 0.000,
                        1.000, 0.333, 0.000,
                        1.000, 0.667, 0.000,
                        1.000, 1.000, 0.000,
                        0.000, 0.333, 0.500,
                        0.000, 0.667, 0.500,
                        0.000, 1.000, 0.500,
                        0.333, 0.000, 0.500,
                        0.333, 0.333, 0.500,
                        0.333, 0.667, 0.500,
                        0.333, 1.000, 0.500,
                        0.667, 0.000, 0.500,
                        0.667, 0.333, 0.500,
                        0.667, 0.667, 0.500,
                        0.667, 1.000, 0.500,
                        1.000, 0.000, 0.500,
                        1.000, 0.333, 0.500,
                        1.000, 0.667, 0.500,
                        1.000, 1.000, 0.500,
                        0.000, 0.333, 1.000,
                        0.000, 0.667, 1.000,
                        0.000, 1.000, 1.000,
                        0.333, 0.000, 1.000,
                        0.333, 0.333, 1.000,
                        0.333, 0.667, 1.000,
                        0.333, 1.000, 1.000,
                        0.667, 0.000, 1.000,
                        0.667, 0.333, 1.000,
                        0.667, 0.667, 1.000,
                        0.667, 1.000, 1.000,
                        1.000, 0.000, 1.000,
                        1.000, 0.333, 1.000,
                        1.000, 0.667, 1.000,
                        0.167, 0.000, 0.000,
                        0.333, 0.000, 0.000,
                        0.500, 0.000, 0.000,
                        0.667, 0.000, 0.000,
                        0.833, 0.000, 0.000,
                        1.000, 0.000, 0.000,
                        0.000, 0.167, 0.000,
                        0.000, 0.333, 0.000,
                        0.000, 0.500, 0.000,
                        0.000, 0.667, 0.000,
                        0.000, 0.833, 0.000,
                        0.000, 1.000, 0.000,
                        0.000, 0.000, 0.167,
                        0.000, 0.000, 0.333,
                        0.000, 0.000, 0.500,
                        0.000, 0.000, 0.667,
                        0.000, 0.000, 0.833,
                        0.000, 0.000, 1.000,
                        0.000, 0.000, 0.000,
                        0.143, 0.143, 0.143,
                        0.286, 0.286, 0.286,
                        0.429, 0.429, 0.429,
                        0.571, 0.571, 0.571,
                        0.714, 0.714, 0.714,
                        0.857, 0.857, 0.857,
                        0.000, 0.447, 0.741,
                        0.50, 0.5, 0
                ]
        ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
