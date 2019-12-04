# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Detection output visualization module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from os.path import isfile, join
import skimage
from skimage import io, filters
import glob

import pycocotools.mask as mask_util

from detectron.utils.colormap import colormap
import detectron.utils.env as envu
import detectron.utils.keypoints as keypoint_utils

# Matplotlib requires certain adjustments in some environments
# Must happen before importing matplotlib
envu.set_up_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator


_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)


#map feature
landscape_coord = {"la_defense":[45,35,'/content/drive/My Drive/map feature/icons/la_defense.jpg'],
          "triomphe":[228,436,'/content/drive/My Drive/map feature/icons/triomphe.jpg'],
          "eiffel_tower":[434,420,'/content/drive/My Drive/map feature/icons/eiffel_tower.jpg'],
          "invalides":[395,580,'/content/drive/My Drive/map feature/icons/invalides.jpg'],
          "place_concord_fountain":[323,635,'/content/drive/My Drive/map feature/icons/de_la_place_concord.jpg'],
          "place_concord_obelisk":[323,635,'/content/drive/My Drive/map feature/icons/de_la_place_concord.jpg'],
          "sacrecoeur":[80,801,'/content/drive/My Drive/map feature/icons/sacrecoeur.jpg'],
          "moulinrouge":[110,719,'/content/drive/My Drive/map feature/icons/moulinrouge.jpg'],
          "palais_garnier":[250,713,'/content/drive/My Drive/map feature/icons/palais_garnier.jpg'],
          "louvre":[380,760,'/content/drive/My Drive/map feature/icons/louvre.jpg'],
          "pompidou":[382,871,'/content/drive/My Drive/map feature/icons/pompidou.jpg'],
          "hotel_de_ville":[429,872,'/content/drive/My Drive/map feature/icons/hotel_de_ville.jpg'],
          "notredame":[471,858,'/content/drive/My Drive/map feature/icons/notredame.jpg'],
          "saint_etienne":[543,839,'/content/drive/My Drive/map feature/icons/saint_etienne.jpg']}

def add_icon_to_map(map_no_icon, icon_classes):
  for icon_class in icon_classes:
    icon_x, icon_y, icon_path = landscape_coord[icon_class]
    icon_x_range, icon_y_range = int(map_no_icon.shape[0]//8), int(map_no_icon.shape[1]//8)
    icon = cv2.resize(cv2.imread(icon_path),(icon_y_range,icon_x_range))
    icon_x = max(0,icon_x-(int(icon_x_range/2)+1))
    if icon_x+icon_x_range > map_no_icon.shape[0]:
      icon_x = map_no_icon.shape[0]-icon_x_range
    icon_y = max(0,icon_y-(int(icon_y_range/2)+1))
    if icon_y+icon_y_range > map_no_icon.shape[1]:
      icon_y = map_no_icon.shape[1]-icon_y_range
    map_no_icon[icon_x:icon_x+icon_x_range,icon_y:icon_y+icon_y_range,:] = icon[:,:,:]

  return map_no_icon

def add_map_to_img(img, icon_classes):
  map_no_icon = cv2.imread('/content/drive/My Drive/map feature/paris_map.jpg')
  map_with_icon = add_icon_to_map(map_no_icon, icon_classes)
  map2 = cv2.resize(map_with_icon, (int(img.shape[1]/4), int(img.shape[0]/4)))
  start_x, start_y = img.shape[0]-map2.shape[0], img.shape[1]-map2.shape[1]
  img[start_x:start_x+map2.shape[0], start_y:start_y+map2.shape[1],:] = map2[:,:,:]
  return img


#filter feature
def split_image_into_channels(image):
    """Look at each image separately"""
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    return red_channel, green_channel, blue_channel

def channel_adjust(channel, values):
    # preserve the original size, so we can reconstruct at the end
    orig_size = channel.shape
    # flatten the image into a single array
    flat_channel = channel.flatten()

    # this magical numpy function takes the values in flat_channel
    # and maps it from its range in [0, 1] to its new squeezed and
    # stretched range
    adjusted = np.interp(flat_channel, np.linspace(0, 255, len(values)), values)

    # put back into the original image shape
    return adjusted.reshape(orig_size)

def sharpen(image, a, b):
    """Sharpening an image: Blur and then subtract from original"""
    blurred = skimage.filters.gaussian(image, sigma=1, multichannel=True)
    sharper = np.clip(image * a - blurred * b, 0, 255)
    return sharper

def merge_channels(red, green, blue):
    """Merge channels back into an image"""
    return np.stack([red, green, blue], axis=2)

def insta_like(image,instafilter):
    # if instafilter == 'gotham':
    #     original_image = skimage.io.imread(image)

    #     original_image = skimage.util.img_as_float(original_image)
    #     r, g, b = split_image_into_channels(original_image)
    #     r_boost_lower = channel_adjust(r, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
    #     r_boost_img = merge_channels(r_boost_lower, g, b)
    #     bluer_blacks = merge_channels(r_boost_lower, g, np.clip(b + 0.03, 0, 1.0))
    #     sharper = sharpen(bluer_blacks, 1.3, 0.3)
    #     r, g, b = split_image_into_channels(sharper)
    #     b_adjusted = channel_adjust(b, [0, 0.047, 0.118, 0.251, 0.318, 0.392, 0.42, 0.439, 0.475, 0.561, 0.58, 0.627, 0.671, 0.733, 0.847, 0.925, 1])
    #     gotham = merge_channels(r, g, b_adjusted) * 255
    #     gotham = gotham.astype(np.uint8)
    #     return gotham

    if instafilter == 'gotham':
        image = cv2.imread(image,1)
        b, g, r = cv2.split(image)
        image = np.stack([r, g, b], axis=2)
        r_boost_lower = channel_adjust(r, [0, 0.05 * 255, 0.1 * 255, 0.2 * 255, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.8 * 255, 0.9 * 255, 0.95 * 255, 1.0 * 255])
        # 2.Make the blacks a little bluer
        bluer_blacks = np.stack([r_boost_lower, g, np.clip(b + 0.1 * 255, 0, 255)], axis=2)
        # 3.A small sharpening
        sharper = sharpen(bluer_blacks, 1.3, 0.3)
        print(sharper[0])
        r, g, b = cv2.split(sharper)
        b_adjusted = channel_adjust(b, [0, 0.047 * 255, 0.118 * 255, 0.251 * 255, 0.318 * 255, 0.392 * 255, 0.42 * 255, 0.439 * 255, 0.475 * 255, 0.561 * 255, 0.58 * 255, 0.627 * 255, 0.671 * 255, 0.733 * 255, 0.847 * 255, 0.925 * 255, 255])
        
        # 4.A boost in blue channel for lower mid-tones and decrease in blue channel for upper mid-tones.
        gotham = np.stack([r, g, b_adjusted], axis=2)
        # gotham = skimage.img_as_ubyte(gotham)
        gotham = gotham.astype(np.uint8)
        return gotham

    elif instafilter == 'vintage':
        im = cv2.imread(image, 1)
        rows, cols = im.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, 300)
        kernel_y = cv2.getGaussianKernel(rows, 300)
        kernel = kernel_y * kernel_x.T
        filter = 255 * kernel / np.linalg.norm(kernel)
        vintage_im = np.copy(im)
        sharper = sharpen(vintage_im, 1.3, 0.3)
        # for each channel in the input image, we will apply the above filter
        for i in range(3):
            vintage_im[:,:,i] = vintage_im[:,:,i] * filter

        vintage_im += 20

        # h, s, v = cv2.split(vintage_im)
        # v += 50
        # vintage_im = cv2.merge((h, s, v))
        return vintage_im.astype(np.uint8)
    
    elif instafilter == 'inkwell':
        #The Greyscale Image of the original one
        inkwell = cv2.imread(image,0)
        #b, g, r = cv2.split(inkwell)
        #inkwell = np.stack([r, g, b], axis=2)
        return inkwell
    
    elif instafilter == 'poprocket':
        poprocket = cv2.imread(image,1)
        poprocket[:,:,2] = poprocket[:,:,2] * 0.18
        poprocket[:,:,1] = poprocket[:,:,1] * 0.1
        #poprocket = poprocket * 0.0140  ##contrast = image * x_contrast
        poprocket = poprocket + 40 ##bright = image_contrast + x_bright
        return poprocket

    elif instafilter == 'walden':
        walden = cv2.imread(image,1)
        walden[:,:,2] = walden[:,:,2] * 1.212
        
        return walden


    elif instafilter == 'canny':
        image = cv2.imread(image,0)
        edges = cv2.Canny(image,100,300)
        return edges
    
    elif instafilter == 'hefe':
        hefe = cv2.imread(image,1)
        hefe[:,:,2] = hefe[:,:,2] * 1
        hefe[:,:,1] = hefe[:,:,1] * 2
        hefe[:,:,0] = hefe[:,:,0] * 1.8
        hefe1 = hefe * 1.0 
        hefe2 = hefe1 + 20
        return hefe2



def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes


def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')


def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""

    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if show_border:
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)


def vis_class(img, pos, class_str, font_scale=0.35):
    """Visualizes the class."""
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, _GREEN, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img


def vis_bbox(img, bbox, thick=1):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
    return img


def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints, _ = keypoint_utils.get_keypoints()
    kp_lines = kp_connections(dataset_keypoints)

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_one_image_opencv(
        im, boxes, segms=None, keypoints=None, thresh=0.9, kp_thresh=2,
        show_box=False, dataset=None, show_class=False):
    """Constructs a numpy array with the detections visualized."""

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return im

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)
        color_list = colormap()
        mask_color_id = 0

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        # show box (off by default)
        if show_box:
            im = vis_bbox(
                im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))

        # show class (off by default)
        if show_class:
            class_str = get_class_string(classes[i], score, dataset)
            im = vis_class(im, (bbox[0], bbox[1] - 2), class_str)
	    im = add_map_to_img(im, class_str)

        # show mask
        if segms is not None and len(segms) > i:
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            im = vis_mask(im, masks[..., i], color_mask)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            im = vis_keypoints(im, keypoints[i], kp_thresh)

     
    # im = insta_like(im, 'hefe')
    im = insta_like(im, 'canny')	
    return im


def vis_one_image(
        im, im_name, output_dir, boxes, segms=None, keypoints=None, body_uv=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext='pdf'):
    """Visual debugging of detections."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return

    dataset_keypoints, _ = keypoint_utils.get_keypoints()

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)

    color_list = colormap(rgb=True) / 255

    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    mask_color_id = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        # show box (off by default)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='g',
                          linewidth=0.5, alpha=box_alpha))

        if show_class:
            ax.text(
                bbox[0], bbox[1] - 2,
                get_class_string(classes[i], score, dataset),
                fontsize=3,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if segms is not None and len(segms) > i:
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = masks[:, :, i]

            _, contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=color_mask,
                    edgecolor='w', linewidth=1.2,
                    alpha=0.5)
                ax.add_patch(polygon)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            plt.autoscale(False)
            for l in range(len(kp_lines)):
                i1 = kp_lines[l][0]
                i2 = kp_lines[l][1]
                if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                    x = [kps[0, i1], kps[0, i2]]
                    y = [kps[1, i1], kps[1, i2]]
                    line = plt.plot(x, y)
                    plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                if kps[2, i1] > kp_thresh:
                    plt.plot(
                        kps[0, i1], kps[1, i1], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

                if kps[2, i2] > kp_thresh:
                    plt.plot(
                        kps[0, i2], kps[1, i2], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

            # add mid shoulder / mid hip for better visualization
            mid_shoulder = (
                kps[:2, dataset_keypoints.index('right_shoulder')] +
                kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            sc_mid_shoulder = np.minimum(
                kps[2, dataset_keypoints.index('right_shoulder')],
                kps[2, dataset_keypoints.index('left_shoulder')])
            mid_hip = (
                kps[:2, dataset_keypoints.index('right_hip')] +
                kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            sc_mid_hip = np.minimum(
                kps[2, dataset_keypoints.index('right_hip')],
                kps[2, dataset_keypoints.index('left_hip')])
            if (sc_mid_shoulder > kp_thresh and
                    kps[2, dataset_keypoints.index('nose')] > kp_thresh):
                x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
                y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                x = [mid_shoulder[0], mid_hip[0]]
                y = [mid_shoulder[1], mid_hip[1]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines) + 1], linewidth=1.0,
                    alpha=0.7)
                
    #   DensePose Visualization Starts!!
    ##  Get full IUV image out 
    IUV_fields = body_uv[1]
    #
    All_Coords = np.zeros(im.shape)
    All_inds = np.zeros([im.shape[0],im.shape[1]])
    K = 26
    ##
    inds = np.argsort(boxes[:,4])
    ##
    for i, ind in enumerate(inds):
        entry = boxes[ind,:]
        if entry[4] > 0.65:
            entry=entry[0:4].astype(int)
            ####
            output = IUV_fields[ind]
            ####
            All_Coords_Old = All_Coords[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2],:]
            All_Coords_Old[All_Coords_Old==0]=output.transpose([1,2,0])[All_Coords_Old==0]
            All_Coords[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2],:]= All_Coords_Old
            ###
            CurrentMask = (output[0,:,:]>0).astype(np.float32)
            All_inds_old = All_inds[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2]]
            All_inds_old[All_inds_old==0] = CurrentMask[All_inds_old==0]*i
            All_inds[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2]] = All_inds_old
    #
    All_Coords[:,:,1:3] = 255. * All_Coords[:,:,1:3]
    All_Coords[All_Coords>255] = 255.
    All_Coords = All_Coords.astype(np.uint8)
    All_inds = All_inds.astype(np.uint8)
    #
    IUV_SaveName = os.path.basename(im_name).split('.')[0]+'_IUV.png'
    INDS_SaveName = os.path.basename(im_name).split('.')[0]+'_INDS.png'
    cv2.imwrite(os.path.join(output_dir, '{}'.format(IUV_SaveName)), All_Coords )
    cv2.imwrite(os.path.join(output_dir, '{}'.format(INDS_SaveName)), All_inds )
    print('IUV written to: ' , os.path.join(output_dir, '{}'.format(IUV_SaveName)) )
    ###
    ### DensePose Visualization Done!!
    #
    output_name = os.path.basename(im_name) + '.' + ext
    fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
    plt.close('all')
