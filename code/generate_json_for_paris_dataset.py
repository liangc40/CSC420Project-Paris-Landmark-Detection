from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import os
import cv2
from PIL import Image
import numpy as np
from detectron.datasets.json_dataset import JsonDataset
example_json = json.load(open('/coco/annotations/person_keypoints_minival2014.json'))
dataset = JsonDataset('paris_2019_train')
dataset_dir = '/coco/paris_dataset'
res_json = json.load(open(os.path.join(dataset_dir, 'paris_building_train.json')))
json_keys = [u'images',u'annotations',u'categories']
im_info = [u'file_name', u'height', u'width', u'id']
ann_info = [u'bbox', u'segmentation', u'num_keypoints', u'area', u'iscrowd', u'image_id', u'category_id', u'id', u'keypoints']
cat_info = []
paris_data = []
classes = {}
folder_idx = 5
ann_info = 6
images_list = []
anns_list = []
new_cat_id = 0
im_id = 0
ann_id = 0
invalid_count = 0
with open('/coco/paris_dataset/train_final.txt') as f:
    for line in f.readlines():
        tmp_im = {}
        tmp_ann = {}
        gt_info = line[:-1].split('/')
        fold_name = os.path.join('paris_data',gt_info[5])
        ann = gt_info[6]
        image_name, str_x1, str_y1, str_x2, str_y2, class_name = ann.split(',')
        image_path = os.path.join(dataset_dir, fold_name, image_name)
        if not os.path.exists(image_path):
            invalid_count += 1
            continue
        im = cv2.imread(image_path)
        if im is None:
            invalid_count += 1
            print('no image:%s'%(image_path))
            continue
        im_h, im_w = im.shape[:2]
        # im info
        tmp_im[u'file_name'] = os.path.join(fold_name, image_name)
        tmp_im[u'height'] = im_h
        tmp_im[u'width'] = im_w
        tmp_im[u'id'] = im_id

        images_list.append(tmp_im)
        im_id += 1
        # ann info
        tmp_ann[u'bbox'] = [float(str_x1), float(str_y1), float(str_x2) - float(str_x1) + 1, float(str_y2) - float(str_y1) + 1]
        assert tmp_ann[u'bbox'][2] >= 0 and tmp_ann[u'bbox'][3] >= 0, 'invalid bbox'
        tmp_ann[u'segmentation'] = []
        tmp_ann[u'keypoints'] = [0]*51
        tmp_ann[u'num_keypoints'] = 0
        tmp_ann[u'iscrowd'] = 0
        tmp_ann[u'area'] = tmp_ann[u'bbox'][2] * tmp_ann[u'bbox'][3]
        if class_name in classes:
            cat_id = classes[class_name]
        else:
            new_cat_id += 1
            cat_id = new_cat_id
            classes[class_name] = new_cat_id
        tmp_ann[u'category_id'] = cat_id
        tmp_ann[u'image_id'] = tmp_im[u'id']
        tmp_ann[u'id'] = ann_id
        anns_list.append(tmp_ann)
        ann_id += 1
        print
for cat in classes.keys():
    cat_info.append({u'supercategory': u'building', u'id': classes[cat], u'name': cat})

paris_json = {}
paris_json[u'images'] = images_list
paris_json[u'annotations'] = anns_list
paris_json[u'categories'] = cat_info
json.dump(paris_json, open(os.path.join(dataset_dir, 'paris_building_train.json'),'w'))
