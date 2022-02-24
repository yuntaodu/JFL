# The code release for paper "Joint Feature and Labeling Function Adaptation for Unsupervised Domain Adaptation".

## Requirement

* Python: 3.6
* PyTorch: 1.5.1 (with suitable CUDA and CuDNN version)
* torchvision: 0.6.1
* tensorboard: 2.3.0
* Scipy
* PIL
* Numpy
* argparse
* easydict

## Datasets:

- Office-31/Office-Home dataset:
  
  You need to modify the path of the image in every ".txt" in "./data".
- Digital dataset:
  
  You need to put the data of each domain in the corresponding folder, with all in one root path.

## Training:

You can run the command in `run.sh` to train and evaluate on each task for digital dataset. Before that, you need to change the `data_root` (data root path), `modeldir`(model saving path), `logdir` (tensorboard saving path) and `cuda` (gpu  options) in the script.

```
python run.py -s mnist -t mnist_m -cuda 0 -logdir /runs/JFL -modeldir /models/JFL -data_root /root/dataset_DA >results/mnist_mnist_m.log  2>&1
```
<!--
## Citation

```
@article{JFL,
  title={Joint Feature and Labeling Function Adaptation for Unsupervised Domain Adaptation},
  year={2020}
}
```
--!>
