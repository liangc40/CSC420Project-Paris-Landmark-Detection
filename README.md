# csc420-project

This is our Paris obejct detection project

This project is based on Facebook detectron research (https://github.com/facebookresearch/Detectron) using FPN(Feature Pyramid Network for Object Detection). 

First of all, set up Inference according to INSTALL.md. (https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md)

Then, create paris_building_train.json file by calling generate_json_for_paris_dataset.py to save ground truth boxes info.

Configure hyperparameters for training in Paris_Detector_ResNet50_FPN_s1x-e2e.yaml. Here we use generalized RCNN, You can change to other models such as Faster RCNN if you want. 

Use CUDA_VISIBLE_DEVICES=9 python2 tools/train_net.py --cfg configs/Paris_Detector_ResNet50_FPN_s1x-e2e.yaml NUM_GPUS 1 OUTPUT_DIR paris_exps/e2e-Res50-fpn-output command to start training.

Save trained weights as model_final.pkl. Because the file is too large, we've already submitted it on Markus.

Use the following command to CUDA_VISIBLE_DEVICES=9 python2 tools/video_infer_simple.py --cfg configs/Paris_Detector_ResNet50_FPN_s1x-e2e.yaml --output-dir paris_exps/e2e-Res50-fpn-output/demo_test_out     --wts paris_exps/e2e-Res50-fpn-output/train/paris_2019_train/generalized_rcnn/model_final.pkl /coco/paris_dataset/test to test videos
