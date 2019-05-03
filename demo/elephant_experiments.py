# imports

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import requests
from io import BytesIO
from PIL import Image
import numpy as np

# this makes our figures bigger
d = .5
pylab.rcParams['figure.figsize'] = 20*d, 12*d
pylab.rc('image', interpolation='bilinear')

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


def main():

    config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda:0"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.5,
    )

    from pycocotools.coco import COCO
    dataType='val2017'
    dataDir='/home/amir/data/coco'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    coco=COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    imgDir = '{}/images/{}'.format(dataDir,dataType)
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))



if __name__ == "__main__":
    main()
