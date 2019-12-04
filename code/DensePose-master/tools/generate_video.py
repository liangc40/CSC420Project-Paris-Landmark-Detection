import cv2
import os
import glob


video_dir = 'paris_exps/e2e-Res50-fpn-output/demo_test_out/PARIS_detector_demo_v1.mp4'
demo_out_dir = 'paris_exps/e2e-Res50-fpn-output/demo_test_out'
im_list = glob.glob(demo_out_dir+'/*.jpg')
num_images = len(im_list)
fourcc = cv2.VideoWriter_fourcc('M','p','4','v')
im_size = (1280, 720)
v_writer = cv2.VideoWriter(video_dir, fourcc, 25, im_size)
for i in range(num_images):
    print('process %d frame'%(i+1))
    im_dir = os.path.join(demo_out_dir, str(i)+'.jpg')
    im = cv2.imread(im_dir)
    im_h, im_w = im.shape[:2]
    if im_w != im_size[0] or im_h != im_size[1]:
        im = cv2.resize(im, im_size)
    v_writer.write(im)

v_writer.release()
