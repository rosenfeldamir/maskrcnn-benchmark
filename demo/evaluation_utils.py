
import numpy as np
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from numpy import argsort,zeros

def populate_ap(metrics_obj,fix_val):
	m = []
	for c in range(1,len(coco_demo.CATEGORIES)):    
		is_found = False
		for z in metrics_obj:
			if z['class'] == c:
				is_found = True
				m.append(dict(class_id=c,ap=z['AP'],fix=fix_val))
				break
		if not is_found:
				m.append(dict(class_id=c,ap=0,fix=fix_val))
	return m

def calc_prec_rec(cur_preds,cur_gt):
	'''
	calculate precision/recall for a set of detections (with scores)
	assuming that they are all of the same class.
	'''
	scores = cur_preds.extra_fields['scores']
	score_order_desc = [argsort(-scores)]
	scores = scores[score_order_desc]
	cur_preds = cur_preds[score_order_desc]
	# now calc the iou score
	ious = boxlist_iou(cur_preds,cur_gt)
	#for r in ious:
	det_thresh = .5
	already_detected = zeros(len(cur_gt))
	if len(cur_preds) == 0:
		return [0],[0],0
	tp,fp = zeros(len(cur_preds)),zeros(len(cur_preds))

	values,inds = ious.max(1)
	for idet,(v,gt_ind) in enumerate(zip(values,inds)):
		if v >=det_thresh:
				if already_detected[gt_ind]:
					fp[idet]=1
					#print('fp')
				else:
					already_detected[gt_ind] = 1
					tp[idet]=1
					#print('tp')
		else:
			fp[idet]=1
			#print('tp')	 
	npos = len(cur_gt)
	acc_FP = np.cumsum(fp)
	acc_TP = np.cumsum(tp)
	rec = acc_TP / npos
	prec = np.divide(acc_TP, (acc_FP + acc_TP))
	return rec,prec,scores
def calc_metrics_single_image(target,pred):
	labels = target.extra_fields['labels'].numpy()
	img_unique_labels = set(labels)
	pred_labels = pred.extra_fields['labels'].numpy()
	res = {}
	for cls in img_unique_labels:
		cur_preds = pred[np.nonzero(pred_labels == cls)[0]]
		cur_gt = target[np.nonzero(labels==cls)[0]]
		rec,prec,scores = calc_prec_rec(cur_preds,cur_gt)
		res[cls] = dict(rec=rec,prec=prec,scores=scores)
	return res,img_unique_labels


def perform_evaluation(results,coco_dataset,coco_eval):
	coco_dt = coco_eval.prepare_for_coco_detection(results,coco_dataset)
	cur_eval_results = coco_eval.evaluate_predictions_on_coco(coco_dataset.coco,coco_dt,'/home/amir/res1.json')
	return cur_eval_results