# PCCA.PyTorch

This repository contains a modified version of the Pointnet2.PyTorch project originally created by Shaoshuai Shi.
![Fig2](https://github.com/user-attachments/assets/b1ba0ffc-4fbb-41a1-9905-ff31fe4aa9f7)
![Fig4](https://github.com/user-attachments/assets/640251fd-9664-44e2-8304-d946de5d8610)

## Original Project

The original project can be found [here](https://github.com/sshaoshuai/Pointnet2.PyTorch).

## Modifications

The following modifications were made to the original code:
- Added attention mechanisms to the PointNet++ architecture.
- Adjusted hyperparameters for specific datasets.
- Refactored and optimized certain parts of the code for better performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



# Pointnet2.PyTorch

* PyTorch implementation of [PointNet++](https://arxiv.org/abs/1706.02413) based on [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).
* Faster than the original codes by re-implementing the CUDA operations. 

## Installation
### Requirements
* Linux (tested on Ubuntu 14.04/16.04)
* Python 3.6+
* PyTorch 1.0

### Install 
Install this library by running the following command:

```shell
cd pointnet2
python setup.py install
cd ../
```

## Examples
Here I provide a simple example to use this library in the task of KITTI ourdoor foreground point cloud segmentation, and you could refer to the paper [PointRCNN](https://arxiv.org/abs/1812.04244) for the details of task description and foreground label generation.

1. Download the training data from [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) website and organize the downloaded files as follows:
```
Pointnet2.PyTorch
├── pointnet2
├── tools
│   ├──data
│   │  ├── KITTI
│   │  │   ├── ImageSets
│   │  │   ├── object
│   │  │   │   ├──training
│   │  │   │      ├──calib & velodyne & label_2 & image_2
│   │  train_and_eval.py
```

2. Run the following command to train and evaluate:
```shell
cd tools
python train_and_eval.py --batch_size 8 --epochs 100 --ckpt_save_interval 2 
```



## Project using this repo:
* [PointRCNN](https://github.com/sshaoshuai/PointRCNN): 3D object detector from raw point cloud.

## Acknowledgement
* [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2): Paper author and official code repo.
* [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch): Initial work of PyTorch implementation of PointNet++. 
