

# Masked Autoencoding for Scalable and Generalizable Decision Making

This is the official implementation for the paper [Masked Autoencoding for Scalable and Generalizable Decision Making
](https://arxiv.org/pdf/2211.12740.pdf).

```
@inproceedings{liu2022masked,
    title={Masked Autoencoding for Scalable and Generalizable Decision Making},
    author={Liu, Fangchen and Liu, Hao and Grover, Aditya and Abbeel, Pieter},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022}
}
```


## Installation
Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 unzip
```

Install dependencies:
```sh
conda env create -f conda_env.yml
conda activate maskdp
```

## Dataset

### Download precollected dataset
We provide the datasets used in the paper on [HuggingFace](https://huggingface.co/datasets/fangchenliu/maskdp_data). Please download there. 
We will update the dataset card soon.


### Collect your own dataset
If you want to customize your own dataset on different environments, please follow the instructions in the ```data_collection``` branch.

## Example Scripts

We provide example scripts in ``train`` and ``eval`` folder to train or evaluate the model. Please modify the path to your local dataset.


## Acknowledgement
* This project is inspired by [ExoRL](https://github.com/denisyarats/exorl). We use the same environment and data collection pipeline.

* The transformer implementation is adapted from [minGPT](https://github.com/karpathy/minGPT) and [original MAE](https://github.com/facebookresearch/mae).

## Contact
If you have any questions, please open an issue or contact fangchen_liu@berkeley.edu.