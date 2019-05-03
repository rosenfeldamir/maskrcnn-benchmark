import numpy as np
from collections import Counter
from vision_utils.boxutils import *

def discard_dets(bb,min_score = .5,discard_type = 'keep_above'):
	if len(bb) > 0:
		scores = bb.extra_fields['scores'].numpy()
		if discard_type == 'keep_above':          
			bb = bb[np.nonzero(scores >= min_score)[0]]
		else:
			bb = bb[np.nonzero(scores < min_score)[0]]
	return bb

def remove_masks(p):
	'''
	Remove all 'mask' extra_fields from set of detections.
	'''
	if 'mask' in p.extra_fields:
		del p.extra_fields['mask']
	return p

def apply_co_occ(new_preds, orig_preds, co_occ, only_novel_classes, min_thresh=.1):
	res = []
	for new,orig in zip(new_preds,orig_preds):
		orig_classes = orig.extra_fields['labels'].numpy()
		new_classes = new.extra_fields['labels'].numpy()

		to_keep = [True]*len(new_classes)
		for i,n in enumerate(new_classes):
			if only_novel_classes and n in orig_classes:
				continue
			p = co_occ[n-1,orig_classes-1]
			if max(p) < min_thresh:
				to_keep[i] = False
		res.append(new[to_keep])
	return res

def get_frequent_labels(bb,min_size = 4):
	labels = bb.extra_fields['labels'].numpy()
	c = Counter(labels)
	freq_labels = []
	for k,v in c.items():
		if v >= min_size:
			freq_labels.append(k)
	return labels, freq_labels

def find_group_detections(bb,min_dets_inside = 3):
	'''
	discard detection that contain multiple other detections on the same kind   
	'''	
	labels,frequent_labels = get_frequent_labels(bb)
	is_group_detection = np.array([False]*len(labels))	
	for L in frequent_labels:
		sel = np.nonzero(labels==L)[0]
		cur_bb = bb[sel]

		boxes = cur_bb.bbox
		areas = bb_areas(cur_bb.bbox)
		ints = bb_int(boxes,boxes)/areas # TODO: make sure dimensions match.

		contains_many = (ints.T > .7).sum(0)	>= min_dets_inside
		is_group_detection[sel[contains_many]] = True					
	return is_group_detection


def merge_all_predictions(origs,news,do_nms=True,min_orig_score=.6):
	bb = []
	num_orig_empty = 0
	num_new_empty = 0  
	for i,(new,orig) in tqdm_notebook(enumerate(zip(news,origs)),total=len(origs),desc='merging'):
		
		if len(new) > 0:
			new = coco_demo.select_top_predictions(new)
		else:
			num_new_empty+=1
		if len(orig) > 0:      
			orig = coco_demo.select_top_predictions(orig)
		else:
			num_orig_empty+=1
		orig = remove_masks(orig)
		if len(new) == 0:
			bb.append(orig)
			continue
		if len(orig) == 0:
			bb.append(new)
			continue
		new = remove_masks(new)
		new = remove_spurious2(new,orig,img,min_orig_score)
		bb.append(merge_detections([new,orig],do_nms=do_nms))
	return bb,num_orig_empty,num_new_empty
def merge_extra_detections(orig,new,keep_below_t=.5,overlap_t=.5,min_orig_score=.3):        
	# 1. Let D be the original detections. These are D_low + D_high
	# 2. Let N be the new ones. For each d in D_low, if there's an overlapping detection n with high confidence
	# in N_high, modify the score of d to that of n.  
	# 3. otherwise, keep the score as it is.
	my_orig = deepcopy(orig)
	#orig_low_conf = discard_dets(orig,keep_below_t,'keep_below')
	new1 = discard_dets(new,.5,discard_type = 'keep_above')
	
	if len(new1) == 0:
		return my_orig,0
	overlaps,candidates = boxlist_iou(new1,my_orig).max(1)
	sel_ = candidates[overlaps>=overlap_t]
	new_scores = new1.extra_fields['scores']
	orig_scores = my_orig.extra_fields['scores']
	n_updated = 0
	for j,(ovp,cnd,old_score,new_score) in enumerate(zip(overlaps,candidates,orig_scores,new_scores)):
		if old_score < keep_below_t and ovp >= overlap_t and old_score>=min_orig_score and new_score > old_score:
			#print(old_score,'==>',new_score)
			orig_scores[j] = new_score
			n_updated+=1
	return my_orig, n_updated
def merge_all_predictions2(origs,news,do_nms=True,min_orig_score=.6):      
	bb = []
	num_orig_empty = 0
	num_new_empty = 0    
	for i,(new,orig) in tqdm_notebook(enumerate(zip(news,origs)),total=len(origs),desc='merging'):
		orig = remove_masks(orig)
		if len(new) == 0 or len(orig) == 0:
			bb.append(orig)
			continue
		#  bb.append(merge_detections([orig],do_nms=do_nms))
		#  continue    
		b,_ = merge_extra_detections(orig,new)    
		bb.append(merge_detections([b],do_nms=do_nms))
	return bb,num_orig_empty,num_new_empty

def remove_spurious2(new_dets,orig_dets,img,min_orig_score=0.6,max_area_in_orig=.7):
	orig_dets1 = discard_dets(orig_dets,min_orig_score)  
	if len(orig_dets1) == 0:
		return new_dets
	M = make_mask_from_boxes(img,orig_dets1)
	new_area_in_orig,new_area = areas_in_mask(new_dets,M)  
	area_ratio = new_area_in_orig/new_area
	keep_me = area_ratio < max_area_in_orig
	new_dets = new_dets[np.nonzero(keep_me)[0]]
	return new_dets
def remove_spurious(new_dets,orig_dets,min_orig_score=0.6):
	orig_dets = discard_dets(orig_dets,min_orig_score)
	if len(orig_dets) == 0:
		return new_dets
	new_area_in_orig = bb_int(new_dets.bbox,orig_dets.bbox)/boxutils.bb_areas(new_dets.bbox).reshape(-1,1)
	new_area_in_orig = new_area_in_orig.max(1)
	max_area_in_orig = .7
	keep_me = new_area_in_orig < max_area_in_orig
	new_dets = new_dets[np.nonzero(keep_me)[0]]
	return new_dets

def merge_detections(dets,do_nms = True):
	my_preds = cat_boxlist(dets)
	my_preds = my_preds[(-my_preds.extra_fields['scores']).argsort()]
	if do_nms:
		my_preds = boxlist_nms(
										my_preds,
										coco_demo.cfg.MODEL.ROI_HEADS.NMS,
										max_proposals=-1,
										score_field="scores")
	return my_preds