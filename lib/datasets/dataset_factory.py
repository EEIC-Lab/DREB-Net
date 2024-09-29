from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path))
sys.path.append(os.path.join(current_path, '..'))
print(sys.path)

from sample.ctdet import CTDetDataset
from lib.datasets.dataset.visdrone2019DET import VisDrone2019DET
from lib.datasets.dataset.uavdt import UAVDT

dataset_factory = {
    'visdrone':VisDrone2019DET,
    'uavdt':UAVDT,
}


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], CTDetDataset):
        pass
    return Dataset
    
