import numpy as np
from scipy.ndimage.morphology import binary_dilation
from vision_utils.boxutils import *

def block_masks(image,masks,dilation_factor=None,fill_color=0):
	if dilation_factor is not None:
		masks = binary_dilation(masks,np.ones((dilation_factor,dilation_factor)))	
	m = np.repeat(np.expand_dims(masks,2),3,2)
	img = (1-m) * fill_color + m*image
	return img.astype(np.uint8)

def mask_from_a_box(mask,box,inflation_factor,is_abs=False):
	#print('m')
	my_bb = inflateBox(box, inflation_factor, is_abs=is_abs)
	plotRectangle_img(mask, my_bb, (1, 1, 1), -1)

def mask_from_boxes(img,preds,args):
	mask = img*0
	# let's block the first object. see what happens.
	for my_bb,bb_score in zip(preds.bbox,preds.extra_fields['scores']):
		if bb_score >= args.masking_thresh:
			mask_from_a_box(mask,my_bb,args.inflation_factor)
	return (mask)

def mask_from_masks(img,preds,args):
	scores = preds.extra_fields['scores']
	mask = preds.extra_fields['mask'].numpy().squeeze(1).copy()
	for i, score in enumerate(scores):
		if score < args.masking_thresh:
			mask[i] *= 0  # discard low confidence masks
	mask = np.repeat(mask.max(0, keepdims=True).transpose(1, 2, 0), 3, 2)
	return mask


def make_mask_from_boxes(img,preds,min_score=0):
	mask = img*0
	# let's block the first object. see what happens.
	for my_bb,bb_score in zip(preds.bbox,preds.extra_fields['scores']):
		if bb_score >= min_score:
			elephant_fix.mask_from_a_box(mask,my_bb,1.0)
	return mask