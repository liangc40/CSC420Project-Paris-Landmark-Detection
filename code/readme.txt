巴黎地标识别算法说明

识别算法基于FPN检测器（可参考论文Feature Pyramid Network for Object Detection），实现平台基于Facebook开源的detectron-1的深度学习框架（caffe2）

整个识别算法的实现，可参考以下流程
1. 环境安装。
- 按照DensePose-master/INSTALL.md 文件搭建实验环境。如果不会的话，可直接用官方提供的docker文件。该文件在Titan X + CUDA-9环境下 测试可行。

2. 数据集准备
- 根据paris_final_data和train_final.txt文件，这里提供了一个转换脚本generate_json_for_paris_dataset.py 用于解析txt文件并转换成项目可用的json数据文件。请参考代码，设置原始文件的路径
并生成相关json文件，以便后面的训练和测试使用。
- 更改文件DensePose-master\detectron\datasets\dataset_catalog.py文件中的paris_2019_train字段下 图像的根路径和上一步生成的json文件的路径
-/materials/dataset/annotations 路径下有存放这次训练的所生成的json文件样例

3.模型训练
- 模型训练需要写一个.yaml配置文件。根据paris_data，这里提供了一个样例DensePose-master\configs\Paris_Detector_ResNet50_FPN_s1x-e2e.yaml。这里采用FPN检测器，如果想换其他检测器
更改相应的backbone和FPN字段，则可以选择Faster_RCNN, RetenaNet, MaskRCNN等检测器框架。
- 训练指令样例：CUDA_VISIBLE_DEVICES=7 python2 tools/train_net.py --cfg configs/Paris_Detector_ResNet50_FPN_s1x-e2e.yaml NUM_GPUS 1 OUTPUT_DIR paris_exps/e2e-Res50-fpn-output
训练完后会自动测试在训练集的AP指标（毕竟数据集没有验证集）。
- /materials/model 里面提供一个已训练好的模型 model_final.pkl, 其在训练集上的精度为92%。

4.视频可视化
- 训练完后，可执行如下命令实现对视频内容的检测：
CUDA_VISIBLE_DEVICES=7 python2 tools/video_infer_simple.py --cfg configs/Paris_Detector_ResNet50_FPN_s1x-e2e.yaml --output-dir paris_exps/e2e-Res50-fpn-output/demo_test_out     --wts paris_exps/e2e-Res50-fpn-output/train/paris_2019_train/generalized_rcnn/model_final.pkl /coco/paris_dataset/test
执行完后，将在路径下paris_exps/e2e-Res50-fpn-output/demo_test_out 输出所有视频帧的检测可视化内容。
再执行脚本tools/generate_video.py完成图像到视频的转换。
*注意：针对以上两个脚本，请修改相应路径后再执行
这里提供对Paris_demo.mp4的检测结果视频，存放在百度网盘
链接：https://pan.baidu.com/s/1tPIw7_o3oaQFZwYIO_qs-A 
提取码：661a 