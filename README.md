# Enhanced MobileNets via Augmentation of Filtering-based Features

## Abstract

Thanks to the significant performance, a MobileNet-based model is one of the potential solutions for mobile applications in image classification. However, taking into account raw image data can lead to a lack of structural information of the input images for the learning process of MobileNets. To this end, we propose to take advantage of structural patterns extracted by a well-known image filter. 

Firstly, for an input image, a popular filter will be applied to capture its geomorphological features. Then, a simple function is executed to concatenate the filtered result with the initial image to form an augmented tensor. Thereby, instead of addressing the raw images, their corresponding augmented tensors will be fed into the learning process of MobileNets to exploit richer features for image description. 

Experimental results on benchmark datasets have validated the effectiveness of our proposal. Particularly, the performance of MobileNetV3 has been boosted by **2.5% on CIFAR-100** and up to **5% on Stanford Dogs**, while the computational complexity remains nearly the same.

---

## Example of Training on Stanford Dogs dataset

### Usage Example

To train a model on the Stanford Dogs dataset with a specified architecture and filter, use the following command:

```bash
python train.py -a=v1 -b=64 -d=dogs --filter='Sobel+Img'
```

### Filter Options

The training script supports applying specific filters to the input images before training. You can choose one of the following filters:

- `'Sobel+Img'`: Applies the Sobel filter.
- `'LoG+Img'`: Applies the Laplacian of Gaussian (LoG) filter.
- `'Gabor+Img'`: Applies the Gabor filter.

### Model Architecture Selection

To specify the MobileNet architecture, the training script includes an argument parser. You can set the architecture using the `-a` or `--architecture` parameter:

```python
parser.add_argument('-a', '--architecture', type=str, default='v1', choices=['v1', 'v2', 'v3'], help='Model architecture name.')
```

Example command to train using MobileNetV3:

```bash
python train.py -a=v3 -b=64 -d=dogs -r='./dataset/' --filter='Sobel+Img'
```

### Dataset Path Configuration

Depending on your system setup, you may need to specify the dataset path. To do this, modify the training script by adding the following argument:

```python
parser.add_argument('-r', '--data-dir', type=str, default='./dataset/', help='Dataset root path.')
```

This allows you to define the dataset path using the `-r` or `--data-dir` parameter when running the script. Example usage:

```bash
python train.py -a=v1 -b=64 -d=dogs -r='./dataset/' --filter='Sobel+Img'
```

## Validate the trained model of MobileNet V1 with the Augmentation of Sobel+Image

This repository includes a pretrained model: best_model.pth, trained using MobileNet V1 with the Sobel + Image augmentation technique.

To evaluate the model on the Stanford Dogs dataset, run:

```bash
python train.py -a=v1 -b=64 -d=dogs --evaluate --filter='Sobel+Img'
```

### Summary

- Use `-a` to specify the model architecture (`v1`, `v2`, `v3`).
- Use `-b` to set the batch size.
- Use `-d` to specify the dataset (`dogs`).
- Use `-r` to set the dataset path.
- Use `--filter` to apply an image filter before training.

This guide ensures smooth configuration and execution of the training process for CIFAR-10.
