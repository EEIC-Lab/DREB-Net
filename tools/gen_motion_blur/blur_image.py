import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import signal
from scipy import misc
from generate_PSF import PSF
from generate_trajectory import Trajectory
from tqdm import tqdm

class BlurImage(object):

    def __init__(self, image_path, PSFs=None, part=None, path__to_save=None):
        """
        :param image_path: path to RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path__to_save: folder to save results.
        """
        if os.path.isfile(image_path):
            self.image_path = image_path
            self.original = misc.imread(self.image_path)
            self.shape = self.original.shape
            if len(self.shape) < 3:
                raise Exception('We support only RGB images.')
        else:
            raise Exception('Not correct path to image.')
        self.path_to_save = path__to_save
        if PSFs is None:
            if self.path_to_save is None:
                self.PSFs = PSF(canvas=max(self.shape[0], self.shape[1])).fit()
            else:
                self.PSFs = PSF(canvas=max(self.shape[0], self.shape[1]), path_to_save=os.path.join(self.path_to_save, 'PSFs.png')).fit(save=True)
        else:
            self.PSFs = PSFs

        self.part = part
        self.result = []


    def blur_image(self, save=False, show=False):
        if self.part is None:
            psf = self.PSFs
        else:
            psf = [self.PSFs[self.part]]
        yN, xN, channel = self.shape
        for p in psf:
            # Ensure PSF is centered and padded to match image dimensions
            psf_padded = np.zeros((yN, xN))
            # Calculate center offset
            center_y, center_x = (yN - p.shape[0]) // 2, (xN - p.shape[1]) // 2
            # Place the PSF at the center of the padded array
            psf_padded[center_y:center_y + p.shape[0], center_x:center_x + p.shape[1]] = p

            # Normalize image
            blured = cv2.normalize(self.original, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # Apply the PSF using convolution for each channel
            for i in range(channel):
                blured[:, :, i] = signal.fftconvolve(blured[:, :, i], psf_padded, 'same')
            # Normalize the blurred image to scale 0 to 255
            blured = cv2.normalize(blured, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
            self.result.append(blured)

        if show or save:
            self.__plot_canvas(show, save)


    def __plot_canvas(self, show, save):
        if len(self.result) == 0:
            raise Exception('Please run blur_image() method first.')
        else:
            plt.close()
            plt.axis('off')
            fig, axes = plt.subplots(1, len(self.result), figsize=(10, 10))
            if len(self.result) > 1:
                for i in range(len(self.result)):
                    axes[i].imshow(cv2.cvtColor(self.result[i], cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(cv2.cvtColor(self.result[0], cv2.COLOR_BGR2RGB))
            if save:
                if self.path_to_save is None:
                    raise Exception('Please specify a path to save.')
                for idx, res in enumerate(self.result):
                    cv2.imwrite(os.path.join(self.path_to_save, self.image_path.split('/')[-1]), res)
            if show:
                plt.show()
                

if __name__ == '__main__':
    folders = ['../datasets/VisDrone-2019-DET/VisDrone2019-DET-train/images',
              '../datasets/VisDrone-2019-DET/VisDrone2019-DET-val/images',
              '../datasets/VisDrone-2019-DET/VisDrone2019-DET-test-dev/images',
    ]

    folder_to_saves = ['../datasets/VisDrone-2019-DET_blur/2_DeblurGAN/blur_image/VisDrone2019-DET-train/images',
                      '../datasets/VisDrone-2019-DET_blur/2_DeblurGAN/blur_image/VisDrone2019-DET-val/images',
                      '../datasets/VisDrone-2019-DET_blur/2_DeblurGAN/blur_image/VisDrone2019-DET-test-dev/images',
    ]


    DONE = 1
    for index in range(3):
        folder = folders[index]
        folder_to_save = folder_to_saves[index]
        os.makedirs(folder_to_save, exist_ok=True)
        params = [0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.002]
        for path in tqdm(os.listdir(folder)):
            DONE = 1
            while DONE:
                try:
                    print(path)
                    expl=np.random.choice(params)
                    print(expl)
                    trajectory = Trajectory(canvas=64, max_len=100, expl=expl).fit()
                    psf = PSF(canvas=64, trajectory=trajectory).fit()
                    BlurImage(os.path.join(folder, path), PSFs=psf,
                            path__to_save=folder_to_save, part=np.random.choice([1, 2, 3])).\
                        blur_image(save=True)
                    DONE = 0
                except:
                    DONE = 1
