

# Masked Autoencoding for Scalable and Generalizable Decision Making

This codebase is a pre-released implementation of [MaskDP](https://openreview.net/forum?id=lNokkSaUbfV).

TODO: clean up codebase,  upload models and datasets.

## Prerequisites

Install [MuJoCo](http://www.mujoco.org/) if it is not already the case:

* Download MuJoCo binaries [here](https://mujoco.org/download).
* Unzip the downloaded archive into `~/.mujoco/`.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.

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

In this paper, we use unsupervised/semi/supervised data collected based on [ExoRL](https://github.com/denisyarats/exorl).
You can follow the repo and collect offline data as described in our appendix. You can also
collect your own offline data. 

## Example Scripts

We provide example in ``example_scripts`` to train or evaluate the model. Please modify the path to your local dataset.