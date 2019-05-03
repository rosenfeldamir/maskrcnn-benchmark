import numpy as np
from vision_utils import boxutils
class Inhibitor:
  def __init__(self):
    pass
    #raise NotImplementedError()
  def make_blocked_image(self,img_orig,predictions):
    '''
    '''    
    #raise NotImplementedError()
    #return blocked_image
    return img_orig.copy()

def evaluate_predictions(predictions,anns):
  '''
  '''
  raise NotImplementedError()


import cv2
def get_blocked_image(image,boxes,inflation=.8,is_abs=False,block_color=0,block_width=-1):
	print('!')
	img = image.copy()
	# let's block the first object. see what happens.
	for my_bb in boxes:
		my_bb = inflateBox(my_bb,inflation,is_abs=is_abs)
		boxutils.plotRectangle_img(img,my_bb,(block_color,block_color,block_color),block_width)
	return img
def remove_context(image,boxes,inflation=1.0,is_abs=False,block_color=1,fill_color=128):
  Z = get_blocked_image(np.zeros_like(image),boxes,inflation=inflation,is_abs=is_abs,block_color=block_color)
  return Z*image+(1-Z)*(np.ones_like(image)*fill_color)
