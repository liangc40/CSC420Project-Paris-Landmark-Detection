����ر�ʶ���㷨˵��

ʶ���㷨����FPN��������ɲο�����Feature Pyramid Network for Object Detection����ʵ��ƽ̨����Facebook��Դ��detectron-1�����ѧϰ��ܣ�caffe2��

����ʶ���㷨��ʵ�֣��ɲο���������
1. ������װ��
- ����DensePose-master/INSTALL.md �ļ��ʵ�黷�����������Ļ�����ֱ���ùٷ��ṩ��docker�ļ������ļ���Titan X + CUDA-9������ ���Կ��С�

2. ���ݼ�׼��
- ����paris_final_data��train_final.txt�ļ��������ṩ��һ��ת���ű�generate_json_for_paris_dataset.py ���ڽ���txt�ļ���ת������Ŀ���õ�json�����ļ�����ο����룬����ԭʼ�ļ���·��
���������json�ļ����Ա�����ѵ���Ͳ���ʹ�á�
- �����ļ�DensePose-master\detectron\datasets\dataset_catalog.py�ļ��е�paris_2019_train�ֶ��� ͼ��ĸ�·������һ�����ɵ�json�ļ���·��
-/materials/dataset/annotations ·�����д�����ѵ���������ɵ�json�ļ�����

3.ģ��ѵ��
- ģ��ѵ����Ҫдһ��.yaml�����ļ�������paris_data�������ṩ��һ������DensePose-master\configs\Paris_Detector_ResNet50_FPN_s1x-e2e.yaml���������FPN�����������뻻���������
������Ӧ��backbone��FPN�ֶΣ������ѡ��Faster_RCNN, RetenaNet, MaskRCNN�ȼ������ܡ�
- ѵ��ָ��������CUDA_VISIBLE_DEVICES=7 python2 tools/train_net.py --cfg configs/Paris_Detector_ResNet50_FPN_s1x-e2e.yaml NUM_GPUS 1 OUTPUT_DIR paris_exps/e2e-Res50-fpn-output
ѵ�������Զ�������ѵ������APָ�꣨�Ͼ����ݼ�û����֤������
- /materials/model �����ṩһ����ѵ���õ�ģ�� model_final.pkl, ����ѵ�����ϵľ���Ϊ92%��

4.��Ƶ���ӻ�
- ѵ����󣬿�ִ����������ʵ�ֶ���Ƶ���ݵļ�⣺
CUDA_VISIBLE_DEVICES=7 python2 tools/video_infer_simple.py --cfg configs/Paris_Detector_ResNet50_FPN_s1x-e2e.yaml --output-dir paris_exps/e2e-Res50-fpn-output/demo_test_out     --wts paris_exps/e2e-Res50-fpn-output/train/paris_2019_train/generalized_rcnn/model_final.pkl /coco/paris_dataset/test
ִ����󣬽���·����paris_exps/e2e-Res50-fpn-output/demo_test_out ���������Ƶ֡�ļ����ӻ����ݡ�
��ִ�нű�tools/generate_video.py���ͼ����Ƶ��ת����
*ע�⣺������������ű������޸���Ӧ·������ִ��
�����ṩ��Paris_demo.mp4�ļ������Ƶ������ڰٶ�����
���ӣ�https://pan.baidu.com/s/1tPIw7_o3oaQFZwYIO_qs-A 
��ȡ�룺661a 