# FilFL (ECAI.24)
## 1. Paper Link
ArXiv: https://arxiv.org/abs/2302.06599

## 2. Dependencies
The code requires Python >= 3.6 and PyTorch >= 1.2.0. To install the other dependencies: `conda env create -f environment.yml`.

## 3. Datasets
This code uses the CIFAR10, Federated Extended MNIST (FEMNIST), Shakespeare datasets.

The CIFAR10 dataset is downloaded automatically by the torchvision package. 
FEMNIST and Shakespeare are provided by the LEAF repository, which should be downloaded from https://github.com/TalwalkarLab/leaf/. 
Then the raw FEMNIST and Shakespeare datasets can be downloaded by following the instructions in LEAF. 

## 4. Training and Testing
Run:
`python3 main.py gpus=[0] output_dir=??? bandit=$1 randomized=$2 datamodule.seed=$3 split.alpha=$4`

Before running, you can change the defaultConf.yaml under the config directory. You can specify the model (e.g resnet_cifar10, resnet_femnist, etc.), the datamodule( e.g, cifar10, shakespeare) and all other parameters. 

## 5. Citateion

Please cite our paper if you use our implementation of FilFL:

```
@article{fourati2023filfl,
  title={FilFL: Client filtering for optimized client participation in federated learning},
  author={Fourati, Fares and Kharrat, Salma and Aggarwal, Vaneet and Alouini, Mohamed-Slim and Canini, Marco},
  journal={arXiv preprint arXiv:2302.06599},
  year={2023}
}
```