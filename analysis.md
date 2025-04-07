# Analysis

Most important files and folders:
- lib/models
- lib/lr_scheduler.py
- lib/ohem_ce_loss.py
- tools/evaluate.py
- tools/train_amp.py

## configs

Contains the possible configurations for the model depending on the dataset used. 

## datasets

Contains the datasets used for training and testing the model. 

## lib

### data

Contains the data loaders for the datasets used in the model.
Dataset loaders:
- base_dataset.py
- cityscapes_cv2.py
- coco.py
- customer_dataset.py
- ade20k.py

- get_dataloader.py\
Gets a config file and returns a dataloader for the dataset specified in the config file.

- sampler.py\
Sampler that restricts data loading to a subset of the dataset. It is useful for distributed training.

- transform_cv2.py\
Contains the data augmentation and transformation functions for the datasets used in the model.

#### Conclusion

Can be replaced with my own implementation for Cityscapes and VOC. 

### models

Contains the model architecture for the BiseNet and BiseNetv2 models.

### Other files

- logger.py\
Contains the logger for the model. It is used to log the training and testing process of the model.
Conclusion: Can be replaced with my own implementation.

- lr_scheduler.py\
Contains the learning rate scheduler for the model. It is used to adjust the learning rate during training.
Types include: WarmupLrScheduler, WarmupPolyLrScheduler, WarmupExpLrScheduler, WarmupCosineLrScheduler, WarmupStepLrScheduler. 
Conclusion: If BiseNetv2 depends on these learning rate schedulers, I should incorporate them into my own implementation.

- meters.py\
Contains TimeMeter and AverageMeter classes.
Conclusion: Can be replaced with my own implementation.

- ohem_ce_loss.py\
Contains the OHEM (online hard example mining) loss function for the model. It is used to calculate the loss during training.\
References: [Code](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/losses/ohem_cross_entropy_loss.py), [Explanation](https://paperswithcode.com/method/ohem)
Conclusion: If BiseNetv2 depends on this loss function, I should incorporate it into my own implementation.

## ncnn

Contains the ncnn implementation of the model. It is used to convert the model to ncnn format for inference on mobile devices.
Conclusion: Can be ignored.

## old

Contains the old implementation of the model. It is used for reference and comparison with the new implementation.
Conclusion: Can be ignored.

## openvino

Contains the openvino implementation of the model. It is used to convert the model to openvino format for inference on mobile devices.
Conclusion: Can be ignored.

## tensorrt

Contains the tensorrt implementation of the model. It is used to convert the model to tensorrt format for inference on mobile devices.
Conclusion: Can be ignored.

## tis

Contains the tis implementation of the model. It is used to convert the model to tis format for inference on mobile devices.
Conclusion: Can be ignored.

## tools

- check_dataset_info.py\
This is a utility for analyzing a dataset of images and their corresponding label images (e.g., for semantic segmentation tasks). It performs several checks and computations to gather useful statistics about the dataset. 
Conclusion: May be useful for reference.

- conver_to_trt.py\
This script is used to convert a PyTorch model to TensorRT format.
Conclusion: Can be ignored.

- demo_video.py\
This file performs semantic segmentation on a video. It processes an input video frame by frame, applies a pre-trained segmentation model to each frame, and saves the segmented output as a new video.
Conclusion: May be useful for reference.

- demo.py\
This file performs semantic segmentation on a single input image using a pre-trained model and saves the segmented output as an image.

- evaluate.py\
This script evaluates the performance of a semantic segmentation model on a given dataset. It computes various metrics such as mean Intersection over Union (mIoU), pixel accuracy, and mean pixel accuracy.
Conclusion: May be useful for reference.

- export_libtorch.py\
This script is used to export a PyTorch model to the LibTorch format, which is the C++ distribution of PyTorch. This allows for inference in C++ applications.
Conclusion: Can be ignored.

- export_onnx.py\
This script is used to export a PyTorch model to the ONNX (Open Neural Network Exchange) format. ONNX is an open format for representing machine learning models, allowing interoperability between different frameworks.
Conclusion: Can be ignored.

- gen_dataset_annos.py\
This script generates annotations for a dataset. It is typically used to create the necessary label files for training and evaluation of segmentation models.
Conclusion: May be useful for reference.

- train_amp.py\

This script is used to train a semantic segmentation model using mixed precision training (AMP). Mixed precision training allows for faster training and reduced memory usage by using both 16-bit and 32-bit floating point numbers.
Conclusion: Extremely useful for reference. I should use this script as a base for my own implementation.