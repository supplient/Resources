#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import matplotlib
matplotlib.use("Agg")
import time

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from PIL import Image
import scipy.misc
import xml.etree.ElementTree as ET
import copy

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5, mark=''):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        #print bbox
        color = 'blue'
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=color, linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(mark+str(time.time())+".jpg")

def generate_attack_image(net,inds,keep,transformer,data_blob,prob_blob, image_name):
    for i in inds:
        print "set diff at "+str(keep[i])
        net.blobs[prob_blob].diff[keep[i],1]=1
        net.blobs[prob_blob].diff[keep[i],15]=-1
	#print cls_boxes[keep[i]]
    
    diffs = net.backward()
    diff_sign_mat = np.sign(diffs[data_blob])
    adversarial_noise = 2.0 * diff_sign_mat
   
    #---------show the noise in the picture
    '''
    data=diff_sign_mat[0]
    np.set_printoptions(threshold='nan') 
    print data
    data[data == 0] = 255
    data[data==-1] = 0
    data[data==1] = 0
    
    x=np.transpose(data,(1,2,0))
    scipy.misc.imsave('1.png',x)
    '''
    #--------------------------
    adv_img = net.blobs[data_blob].data[0] + adversarial_noise[0]
    print "noise*100000:"
    print np.max(adversarial_noise[0]*100000)
    attack_hwc = transformer.deprocess(data_blob, adv_img)
    attack_hwc[attack_hwc > 1] = 1.
    attack_hwc[attack_hwc < 0] = 0.
    
    result=attack_hwc*255
    img = Image.fromarray(result.astype('uint8')) # convert image to uint8s
    img.save(os.path.join(cfg.DATA_DIR, 'demo_02', image_name))

def load_pascal_annotation(image_name):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    filename = '../data/VOCdevkit/VOC2007/Annotations/' + image_name + '.xml'
    tree = ET.parse(filename)
    objs = tree.findall('object')
    # Exclude the samples labeled as difficult
    non_diff_objs = [
        obj for obj in objs if int(obj.find('difficult').text) == 0]
    # if len(non_diff_objs) != len(objs):
    #     print 'Removed {} difficult objects'.format(
    #         len(objs) - len(non_diff_objs))
    objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
   
    # Load object bounding boxes into a data frame.

    for ix, obj in enumerate(objs):
        if obj.find('name').text.lower().strip() == 'person':
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes[ix, :] = [x1, y1, x2, y2]
    remove = np.where(boxes[:,2] == 0)
    boxes = np.delete(boxes,remove,axis=0)
    return boxes

def judge_iou(BBGT,BB):
    inds = np.zeros((BB.shape[0],1),dtype=np.uint16)
    for d in range(BB.shape[0]): #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        bb = BB[d, :].astype(float)
        ovmax = -np.inf

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > 0.5: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
            inds[d,0] = 1
    return inds

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    
    data_blob='data'
    prob_blob='cls_prob'
    
    channel_means = np.array([102.9801, 115.9465, 122.7717])
    transformer = caffe.io.Transformer({data_blob: net.blobs[data_blob].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', channel_means)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    # Load the demo image
    print "Loading image"
    im_file = os.path.join(cfg.DATA_DIR, 'demo_01', image_name)
    im = cv2.imread(im_file)
    timer = Timer()
    timer.tic()
    #scores, boxes, im_scale = im_detect(net, im,roi)
    #scale = copy.deepcopy(im_scale)
    print "Detecting..."
    scores, boxes = im_detect(net, im)
    #scale = net.blobs['im_info'].data[0][2] #for calculate the l2 norm
    timer.toc()
    print ('Detection took {:.3f}s for '
          '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    for ii in range(20) :
        # Detect all object classes and regress object bounds
        #print ('Detection took {:.3f}s for '
        #       '{:d} object proposals').format(timer.total_time, boxes.shape[0])
        #-------------------lc
	print "Start "+str(ii)+" turn."
        
        net.blobs[prob_blob].diff[...] = 0

    	max_scores_ind = np.argmax(scores[:, 1:], 1)+1# means the ith box's max_scores_ind[i] class has highest score, and skip the background(+1)
    	max_box_ind = np.argmax(max_scores_ind)# means the max_box_ind th box has highest class score for all class
    	max_score_ind = max_scores_ind[max_box_ind]# the max_box_ind th box's highest score's class
    	max_box = boxes[max_box_ind, 4*max_score_ind:4*(max_score_ind+1)]
    	max_score = scores[max_box_ind, max_score_ind]
    
        cls_boxes = boxes[:, 4*15:4*(15 + 1)]#the 'people' class
        cls_scores = scores[:, 15]
        dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, 0.3)
        dets = dets[keep, :]
        
        #judge the bbox with the GT iou whether > 0.5
        vis_detections(im, 'person', dets, thresh=0.3, mark="Fir+"+str(ii)+"+")

        print "Origin dets:"
        print dets.shape
        print dets
        inds = np.where(dets[:, -1] >= 0.1)[0]
        print dets[:, -1]
        print len(inds)
        keep = np.array(keep)
        #print keep.shape
        #print keep2
        #print keep[keep2]
        if (len(inds)>0) or max_score<0.5 :
            print 'attack....'
            generate_attack_image(net, inds, keep, transformer, data_blob, prob_blob, image_name)
        else :
            break

        im_file = os.path.join(cfg.DATA_DIR, 'demo_02', image_name)
        im = cv2.imread(im_file)
        timer.tic()
        scores, boxes = im_detect(net, im)
        timer.toc()

        

    # Visualize detections for each class
    
#    CONF_THRESH = 0.0
#    NMS_THRESH = 0.3
    max_scores_ind = np.argmax(scores[:, 1:], 1)+1# means the ith box's max_scores_ind[i] class has highest score, and skip the background(+1)
    max_box_ind = np.argmax(max_scores_ind)# means the max_box_ind th box has highest class score for all class
    max_score_ind = max_scores_ind[max_box_ind]# the max_box_ind th box's highest score's class
    max_box = boxes[max_box_ind, 4*max_score_ind:4*(max_score_ind+1)]
    max_score = scores[max_box_ind, max_score_ind]
    max_box_array = max_box[np.newaxis, :]
    max_score_array = np.array([max_score])[np.newaxis, :]
    dets = np.hstack((max_box_array, max_score_array)).astype(np.float32)
    print "Final dets:"
    print dets
    vis_detections(im, CLASSES[max_score_ind], dets, thresh=0.0, mark="Sec")
    
    #for cls_ind, cls in enumerate(CLASSES[1:]):
    #    cls_ind += 1 # because we skipped background
    #    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    #    cls_scores = scores[:, cls_ind]
    #    dets = np.hstack((cls_boxes,
    #                      cls_scores[:, np.newaxis])).astype(np.float32)
    #    keep = nms(dets, NMS_THRESH)
    #    dets = dets[keep, :]
#
#
#        if cls == 'person':
#            vis_detections(im, cls, dets, thresh=CONF_THRESH, mark="Sec")
    

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args




if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    #cfg.TEST.BBOX_REG = False
    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                               NETS[args.demo_net][1])
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)


    #file_r = open('../data/scale.txt')  
    im_names = ['233']#001551
    
    for im_name in im_names:
        #scale = file_r.readline()
        scale = 0#1.69971671388 
        im_name = im_name + '.jpg'
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo_01/{}'.format(im_name)
      
        demo(net, im_name)

    #file_r.close()
    plt.show()
