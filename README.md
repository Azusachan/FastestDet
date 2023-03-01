# How to use
## Dependent installation
* PiP(Note pytorch CUDA version selection)
  ```
  pip install -r requirements.txt
  ```
## Test
* Picture test
  ```
  python3 test.py --yaml configs/coco.yaml --weight weights/weight_AP05:0.253207_280-epoch.pth --img data/3.jpg
  ```
<div align=center>
<img src="https://github.com/dog-qiuqiu/FastestDet/blob/main/result.png"> />
</div>

## How to train
### Building data sets(The dataset is constructed in the same way as darknet yolo)
* The format of the data set is the same as that of Darknet Yolo, Each image corresponds to a .txt label file. The label format is also based on Darknet Yolo's data set label format: "category cx cy wh", where category is the category subscript, cx, cy are the coordinates of the center point of the normalized label box, and w, h are the normalized label box The width and height, .txt label file content example as follows:
  ```
  11 0.344192634561 0.611 0.416430594901 0.262
  14 0.509915014164 0.51 0.974504249292 0.972
  ```
* The image and its corresponding label file have the same name and are stored in the same directory. The data file structure is as follows:
  ```
  .
  ├── train
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  └── val
      ├── 000043.jpg
      ├── 000043.txt
      ├── 000057.jpg
      ├── 000057.txt
      ├── 000070.jpg
      └── 000070.txt
  ```
* Generate a dataset path .txt file, the example content is as follows：
  
  train.txt
  ```
  /home/qiuqiu/Desktop/dataset/train/000001.jpg
  /home/qiuqiu/Desktop/dataset/train/000002.jpg
  /home/qiuqiu/Desktop/dataset/train/000003.jpg
  ```
  val.txt
  ```
  /home/qiuqiu/Desktop/dataset/val/000070.jpg
  /home/qiuqiu/Desktop/dataset/val/000043.jpg
  /home/qiuqiu/Desktop/dataset/val/000057.jpg
  ```
* Generate the .names category label file, the sample content is as follows:
 
  category.names
  ```
  person
  bicycle
  car
  motorbike
  ...
  
  ```
* The directory structure of the finally constructed training data set is as follows:
  ```
  .
  ├── category.names        # .names category label file
  ├── train                 # train dataset
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  ├── train.txt              # train dataset path .txt file
  ├── val                    # val dataset
  │   ├── 000043.jpg
  │   ├── 000043.txt
  │   ├── 000057.jpg
  │   ├── 000057.txt
  │   ├── 000070.jpg
  │   └── 000070.txt
  └── val.txt                # val dataset path .txt file

  ```
### Build the training .yaml configuration file
* Reference./configs/coco.yaml
  ```
  DATASET:
    TRAIN: "/home/qiuqiu/Desktop/coco2017/train2017.txt"  # Train dataset path .txt file
    VAL: "/home/qiuqiu/Desktop/coco2017/val2017.txt"      # Val dataset path .txt file 
    NAMES: "dataset/coco128/coco.names"                   # .names category label file
  MODEL:
    NC: 80                                                # Number of detection categories
    INPUT_WIDTH: 352                                      # The width of the model input image
    INPUT_HEIGHT: 352                                     # The height of the model input image
  TRAIN:
    LR: 0.001                                             # Train learn rate
    THRESH: 0.25                                          # ？？？？
    WARMUP: true                                          # Trun on warm up
    BATCH_SIZE: 64                                        # Batch size
    END_EPOCH: 350                                        # Train epichs
    MILESTIONES:                                          # Declining learning rate steps
      - 150
      - 250
      - 300
  ```
### Train
* Perform training tasks
  ```
  python3 train.py --yaml configs/coco.yaml
  ```
### Evaluation
* Calculate map evaluation
  ```
  python3 eval.py --yaml configs/coco.yaml --weight weights/weight_AP05:0.253207_280-epoch.pth
  ```
* COCO2017 evaluation
  ```
  creating index...
  index created!
  creating index...
  index created!
  Running per image evaluation...
  Evaluate annotation type *bbox*
  DONE (t=30.85s).
  Accumulating evaluation results...
  DONE (t=4.97s).
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.130
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.253
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.119
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.021
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.129
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.237
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.142
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.208
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.214
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.043
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.236
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.372

  ```
# Deploy
## Export onnx
* You can export .onnx by adding the --onnx option when executing test.py
  ```
  python3 test.py --yaml configs/coco.yaml --weight weights/weight_AP05:0.253207_280-epoch.pth --img data/3.jpg --onnx
  ```
## Export torchscript
* You can export .pt by adding the --torchscript option when executing test.py
  ```
  python3 test.py --yaml configs/coco.yaml --weight weights/weight_AP05:0.253207_280-epoch.pth --img data/3.jpg --torchscript
  ```
## NCNN
* Need to compile ncnn and opencv in advance and modify the path in build.sh
  ```
  cd example/ncnn/
  sh build.sh
  ./FastestDet
  ```
## onnx-runtime
* You can learn about the pre and post-processing methods of FastestDet in this Sample
  ```
  cd example/onnx-runtime
  pip install onnx-runtime
  python3 runtime.py
  ```
# Citation
* If you find this project useful in your research, please consider cite:
  ```
  @misc{=FastestDet,
        title={FastestDet: Ultra lightweight anchor-free real-time object detection algorithm.},
        author={xuehao.ma},
        howpublished = {\url{https://github.com/dog-qiuqiu/FastestDet}},
        year={2022}
  }
  ```
# Reference
* https://github.com/Tencent/ncnn
