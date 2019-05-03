from copy import deepcopy
import matplotlib.pylab as pylab
import seaborn as sns
import numpy as np
import argparse
import os
from tqdm import tqdm
import torch
import pickle
from glob import glob
#from inhibitor import *
import skimage.transform
import elephant_fix
from pycocotools.coco import COCO
from numpy import array, argsort, zeros

from vision_utils.boxutils import *
from vision_utils.generic_utils import dict_to_param_str, str2bool, find
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

from postprocessing import *
from mask_ops import *

# this makes our figures bigger
d = .5
pylab.rcParams['figure.figsize'] = 20 * d, 12 * d
pylab.rc('image', interpolation='bilinear')

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.coco import COCODataset
from predictor import COCODemo
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from matplotlib.pyplot import imshow
from maskrcnn_benchmark.data.datasets.evaluation.coco import coco_eval
from maskrcnn_benchmark.data.datasets.coco import COCODataset

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda:0"])


def re_detect2(img, preds, min_score=0, max_score=1, inflation_factor=1, is_abs=False,
               min_keep_iou=.5):
    scores = preds.extra_fields['scores']
    labels = preds.extra_fields['labels'].numpy()
    masking_color = 128
    masked_images = []
    new_dets_map = {}
    for ibox, score in tqdm(enumerate(scores), total=len(preds)):
        bb = preds[[ibox]]
        if score > max_score or score < min_score:
            continue
        cur_label = bb.extra_fields['labels'][0]
        cur_mask = img * 0
        mask_from_a_box(cur_mask, bb.bbox[0], inflation_factor, is_abs=is_abs)
        masked_img = img * cur_mask + (1 - cur_mask) * masking_color
        preds_masked = coco_demo.compute_prediction(masked_img[:, :, ::-1])
        remove_masks(preds_masked)
        if len(preds_masked) > 0:
            ious = boxlist_iou(bb, preds_masked)
            ious_above_thresh = find((ious >= min_keep_iou).numpy().ravel())
            # throw away low-iou matches
            preds_masked = preds_masked[find(
                (ious >= min_keep_iou).numpy().ravel())]
            if len(preds_masked) > 0:
                new_dets_map[ibox] = dict(source_ind=ibox,
                                          source=bb, result=preds_masked,
                                          result_ious=ious[0,
                                                           ious_above_thresh],
                                          min_score=min_score,
                                          max_score=max_score, inflation_factor=inflation_factor,
                                          is_abs=is_abs, min_keep_iou=min_keep_iou)
    return new_dets_map


def merge_extra_detections2(preds, new_dets_map, filter_by_label=False, ignore_orig_score_above=1,
                            ignore_orig_score_below=0):
    scores = preds.extra_fields['scores']
    new_preds = deepcopy(preds)
    n_updated = 0
    n0 = 0
    n1 = 0
    n2 = 0
    for ibox, score in tqdm(enumerate(scores), total=len(preds)):
        if ibox not in new_dets_map:
            continue
        n0 += 1
        b = new_dets_map[ibox]
        res = b['result']
        src = b['source']
        res_labels = res.extra_fields['labels']
        src_labels = src.extra_fields['labels']
        if filter_by_label:
            is_same_label = find((res_labels == src_labels).numpy().ravel())
            if len(is_same_label) == 0:
                continue
            res = res[is_same_label]
        if len(res) == 0:
            continue
        n2 += 1
        cur_score = src.extra_fields['scores']

        if cur_score > ignore_orig_score_above or cur_score < ignore_orig_score_below:
            continue
        m = res.extra_fields['scores'].max()
        if m > cur_score:
            new_preds.extra_fields['scores'][ibox] = m
            n_updated += 1
    return new_preds, n_updated


def predict_and_show(v):
    image = U.load_image(v)
    img_orig = image.copy()
    dd = 1
    r = 1
    fix, ax = plt.subplots(2, 1, figsize=(12 * dd, r * 20 * dd))
    predictions = coco_demo.compute_prediction(image[:, :, ::-1])
    coco_demo.confidence_threshold = .2
    top_predictions = coco_demo.select_top_predictions(predictions)
    img_w_boxes = coco_demo.overlay_boxes(image, top_predictions, box_width=2)
    img_w_boxes_and_names = coco_demo.overlay_class_names(
        image, top_predictions)
    ax.flat[1].imshow(img_orig)
    coco.showAnns(U.get_anns(v))
    ax.flat[0].imshow(img_w_boxes_and_names)
    [r.axis('off') for r in ax.flat]
    return img_orig, top_predictions


def remove_masks(p):
    if 'mask' in p.extra_fields:
        del p.extra_fields['mask']
    return p


def remove_masks_file(fname):
    p = pickle.load(open(fname, 'rb'))

    if type(p['preds']) is list and len(p['preds']) == 0:
        p['preds'] = BoxList(torch.Tensor(0, 4), (300, 300))
        pickle.dump(p, open(fname, 'wb'))
    if 'mask' in p['preds'].extra_fields:
        del p['preds'].extra_fields['mask']
        pickle.dump(p, open(fname, 'wb'))
    return p


def detect_masked(img, coco_demo, preds, args, masking_color=128):
    masking_thresh = args.masking_thresh
    if args.all_at_once:
        if args.masking_type == 'bbox':
            mask = mask_from_boxes(img, preds, args)
            # mask = get_blocked_image(img,preds.bbox,inflation=masking_args.inflation_factor,is_abs=False,
            # block_color=masking_color)
        else:
            # mask = mask_from_masks(img,preds) # inflation factor not used here.
            if len(preds) == 0:
                mask = img * 0
            else:
                mask = mask_from_masks(img, preds, args)
        if args.negative_masking:
            mask = 1 - mask
        masked_img = img * mask + (1 - mask) * masking_color
        preds_masked = coco_demo.compute_prediction(masked_img[:, :, ::-1])
        return preds_masked
    else:
        all_preds = []
        ff = preds.extra_fields
        # if 'mask' in preds.extra_fields:
        # del preds.extra_fields['mask']
        n_above = sum(ff['scores'].numpy() >= args.masking_thresh)
        for bb, score, mask in tqdm(zip(preds.bbox, ff['scores'], ff['mask']), desc='masking one by one...', total=n_above):
            if score < args.masking_thresh:
                continue
            if args.masking_type == 'bbox':
                cur_mask = img * 0
                mask_from_a_box(cur_mask, bb, args.inflation_factor)
            else:
                cur_mask = np.repeat(mask.numpy().transpose(1, 2, 0), 3, 2)
            if args.negative_masking:
                cur_mask = 1 - cur_mask
            masked_img = img * cur_mask + (1 - cur_mask) * masking_color

            preds_masked = coco_demo.compute_prediction(masked_img[:, :, ::-1])

            all_preds.append(preds_masked)
            if 'mask' in preds_masked.extra_fields:
                del preds_masked.extra_fields['mask']
        if len(all_preds) > 0:
            all_preds = cat_boxlist(all_preds)
            all_preds = boxlist_nms(
                all_preds,
                .5,
                max_proposals=100,
                score_field="scores")
        else:
            all_preds = BoxList(torch.Tensor(0, 4), img.shape[:2])
        return all_preds


def initial_detection(coco_dataset, coco_demo, args, max_images=10**9, savedir=None):
    D = os.path.join(savedir, 'orig')
    os.makedirs(D, exist_ok=True)
    pickle.dump(args, open(os.path.join(D, 'args.pkl'), 'wb'))
    N = min(len(coco_dataset), max_images)

    # for i, (img, target, idx) in tqdm(enumerate(coco_dataset), total=N):
    for i in tqdm(range(N)):
        if i == max_images:
            break
        idx = i

        k = coco_dataset.id_to_img_map[idx]
        out_path = os.path.join(D, '{}.pkl'.format(k))
        if os.path.isfile(out_path):
            continue
        img, target, idx_ = coco_dataset[i]
        assert(idx_ == idx)
        image = np.array(img)
        with torch.no_grad():
            img = image.copy()
            preds = coco_demo.compute_prediction(img[:, :, ::-1])
            # if 'mask' in preds.extra_fields:
            # del preds.extra_fields['mask']
        pickle.dump(dict(idx=idx, k=k, preds=preds), open(out_path, 'wb'))


def masked_detection(coco_dataset, coco_demo, args):

    savedir = args.save_dir
    D = os.path.join(savedir, 'orig')

    masking_args = dict(masking_type=args.masking_type, negative_masking=args.negative_masking,
                        inflation_factor=args.inflation_factor, all_at_once=args.all_at_once, masking_thresh=args.masking_thresh)

    out_D = os.path.join(savedir, dict_to_param_str(masking_args))
    os.makedirs(out_D, exist_ok=True)
    pickle.dump(args, open(os.path.join(out_D, 'args.pkl'), 'wb'))

    orig_res_paths = glob(D + '/*.pkl')
    print('!!!!!!!!!!!!!!')
    print(out_D)

    N = min(len(coco_dataset), args.max_images)

    # for i,(img,target,idx) in tqdm(enumerate(coco_dataset),total=N):
    for i in tqdm(range(N)):
        if i == args.max_images:
            break
        idx = i
        k = coco_dataset.id_to_img_map[idx]
        out_path = os.path.join(out_D, '{}.pkl'.format(k))
        if os.path.isfile(out_path):
            continue
        p = os.path.join(D, '{}.pkl'.format(k))
        #cur_res = remove_masks(p)
        cur_res = pickle.load(open(p, 'rb'))
        idx, k_, pred = cur_res['idx'], cur_res['k'], cur_res['preds']
        img, target, idx = coco_dataset[i]
        image = np.array(img)
        preds = detect_masked(image, coco_demo, pred, args, masking_color=128)
        if len(preds) > 0:
            if 'mask' in preds.extra_fields:
                del preds.extra_fields['mask']
        pickle.dump(dict(idx=idx, k=k, preds=preds), open(out_path, 'wb'))


def masked_detection_2(coco_dataset, coco_demo, args):
    savedir = args.save_dir
    D = os.path.join(savedir, 'orig')

    masking_args = dict(masking_type=args.masking_type, negative_masking=args.negative_masking,
                        inflation_factor=args.inflation_factor, all_at_once=args.all_at_once,
                        masking_thresh=args.masking_thresh, inflation_is_abs=args.inflation_is_abs)

    out_D = os.path.join(savedir, dict_to_param_str(masking_args))
    os.makedirs(out_D, exist_ok=True)
    pickle.dump(args, open(os.path.join(out_D, 'args.pkl'), 'wb'))

    print(out_D)

    N = min(len(coco_dataset), args.max_images)
    # for i,(img,target,idx) in tqdm(enumerate(coco_dataset),total=N):
    for i in tqdm(range(N)):
        if i == args.max_images:
            break
        idx = i
        k = coco_dataset.id_to_img_map[idx]
        out_path = os.path.join(out_D, '{}.pkl'.format(k))
        if os.path.isfile(out_path):
            continue
        p = os.path.join(D, '{}.pkl'.format(k))
        cur_res = pickle.load(open(p, 'rb'))

        idx, k_, pred = cur_res['idx'], cur_res['k'], cur_res['preds']
        pred = remove_masks(pred)
        img, target, idx = coco_dataset[i]
        img = np.array(img)
        new_dets_map = re_detect2(img, pred, min_score=0, max_score=1, inflation_factor=args.inflation_factor,
                                  is_abs=args.inflation_is_abs, min_keep_iou=.5)

        pickle.dump(new_dets_map, open(out_path, 'wb'))


def areas_in_mask(preds, M):
    M_int = skimage.transform.integral_image(M[:, :, 0])
    bb = preds.bbox.numpy().astype(int)
    starts = [(b, a) for a, b in bb[:, :2]]
    ends = [(b, a) for a, b in bb[:, 2:]]
    return skimage.transform.integrate(M_int, starts, ends), bb_areas(bb)


def get_label_names(preds):
    return [coco_demo.CATEGORIES[q] for q in top_predictions.extra_fields['labels']]


sns.set_context('paper', font_scale=1.5)


def re_detect(img, preds, min_score=0, max_score=1, inflation_factor=1, is_abs=False):
    scores = preds.extra_fields['scores']
    masking_color = 128
    all_preds = []
    masked_images = []
    target_boxes = []
    # for bb,score in tqdm_notebook(zip(preds.bbox,scores),total=len(preds)):
    for bb, score in zip(preds.bbox, scores):
        if score >= max_score or score <= min_score:
            continue
        cur_mask = img * 0
        mask_from_a_box(cur_mask, bb, inflation_factor, is_abs=is_abs)
        #cur_mask = 1-cur_mask
        masked_img = img * cur_mask + (1 - cur_mask) * masking_color
        preds_masked = coco_demo.compute_prediction(masked_img[:, :, ::-1])
        remove_masks(preds_masked)
        if len(preds_masked) > 0:
            all_preds.append(preds_masked)
        masked_images.append(masked_img)
        target_boxes.append(bb)
    return all_preds, masked_images, target_boxes


config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda:0"])
cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 1000
coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5)


def make_parser():
    parser = argparse.ArgumentParser(description='Elephant In the Room Fixes')
    parser.add_argument('-dataType', default='val2017')
    parser.add_argument('-dataDir', default='/home/amir/data/coco')
    parser.add_argument('-max_images', type=int, default=10,
                        help='maximal number of images')
    parser.add_argument('-masking_type', default='bbox', choices=['bbox', 'mask'],
                        help='mask using bounding box (or "mask" to use the predicted mask)')
    parser.add_argument('-masking_thresh', type=float, default=0.5,
                        help='minimal score for detection to be used as a mask')
    parser.add_argument("-inflation_is_abs", type=str2bool, default=False,
                        help="use absolute instead of relative inflation for bounding boxes")
    parser.add_argument('-negative_masking', action='store_true',
                        help='mask only the detection (otherwise mask everything but the detection)')

    parser.add_argument('-all_at_once', action='store_false',
                        help='mask all detections then run (otherwise one by one))')

    parser.add_argument('-inflation_factor', type=float, default=1.0,
                        help='how much to change size (multiplicative) of bbox/mask before masking')
    parser.add_argument('-save_dir', default='/home/amir/elephant_fix_results')
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    # loading the mscoco validation set .

    dataType = args.dataType
    dataDir = args.dataDir
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    #nms=[cat['name'] for cat in cats]
    imgDir = '{}/images/{}'.format(dataDir, dataType)
    #print('COCO categories: \n{}\n'.format(' '.join(nms)))
    #nms = set([cat['supercategory'] for cat in cats])
    #print('COCO supercategories: \n{}'.format(' '.join(nms)))

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.5)

    coco_dataset = COCODataset(
        annFile, dataDir + '/images/' + args.dataType, True)
    initial_detection(coco_dataset, coco_demo, args,
                      max_images=args.max_images, savedir=args.save_dir)
    print(args)
    #masked_detection( coco_dataset, coco_demo, args)
    masked_detection_2(coco_dataset, coco_demo, args)

    print('all_at_once:', args.all_at_once)


if __name__ == "__main__":
    main()
