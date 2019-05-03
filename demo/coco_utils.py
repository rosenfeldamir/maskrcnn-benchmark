from matplotlib import pyplot as plt
from os.path import join as osj
import numpy as np
class coco_utils:
  def __init__(self, coco, imgDir):
    self.coco = coco
    self.imgDir = imgDir
  def load_image(self, v):
    coco = self.coco        
    imgPath = osj(self.imgDir,v['file_name'])
    img = plt.imread(imgPath)
    if len(img.shape)==2:
      img = np.repeat(np.expand_dims(img,2),3,2)
    return img
  def get_anns(self, v):
    coco = self.coco
    anns = [coco.anns[r] for r in coco.getAnnIds(v['id'])]
    return anns
  def show_anns(self, v): # v is an image record
    coco = self.coco
    img = self.load_image(v)
    plt.imshow(img,interpolation='bilinear')
    anns = self.get_anns(v)
    coco.showAnns(anns)