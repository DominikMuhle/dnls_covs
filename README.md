# Learning Correspondence Uncertainty via Differentiable Nonlinear Least Squares

[**Paper**](https://arxiv.org/abs/2305.09527) |  [**Video** (Soon)](#) | [**Project Page**](https://dominikmuhle.github.io/dnls_covs/)

This is the official implementation for the CVPR 2023 paper:

> **Learning Correspondence Uncertainty via Differentiable Nonlinear Least Squares**
>
> [Dominik Muhle](https://vision.in.tum.de/members/muhled)<sup>1,2</sup>, [Lukas Koestler](https://lukaskoestler.com)<sup>1,2</sup>, [Krishna Jatavallabhula](https://krrish94.github.io)<sup>4</sup> and [Daniel Cremers](https://vision.in.tum.de/members/cremers)<sup>1,2,3</sup><br>
> <sup>1</sup>Technical University of Munich, <sup>2</sup>Munich Center for Machine Learning, <sup>3</sup>University of Oxford, <sup>4</sup>MIT
> 
> [**CVPR 2023** (arXiv)](https://arxiv.org/abs/2305.09527)

# Abstract

We propose a differentiable nonlinear least squares framework to account for uncertainty in relative pose estimation from feature correspondences. Specifically, we introduce a symmetric version of the probabilistic normal epipolar constraint, and an approach to estimate the covariance of feature positions by differentiating through the camera pose estimation procedure. We evaluate our approach on synthetic, as well as the KITTI and EuRoC real-world datasets. On the synthetic dataset, we confirm that our learned covariances accurately approximate the true noise distribution. In real world experiments, we find that our approach consistently outperforms state-of-the-art non-probabilistic and probabilistic approaches, regardless of the feature extraction algorithm of choice.

# Overview

![Teaser Figure](https://dominikmuhle.github.io/dnls_covs/assets/teaser.png)

a) We propose a symetric extension of the [Probabiltistic Normal Epipolar Constraint](https://arxiv.org/abs/2204.02256) (PNEC) to more accurately model the geometry of relative pose estimation with uncertain feature positions.

b) We propose a learning strategy to minimize the relative pose error by learning feature position uncertainty through differentiable nonlinear least squares (DNLS). This learning strategy can be combined with any feature extraction algorithm. We evaluate our learning framework with synthetic experiments and on real-world data in a visual odometry setting.  We show that our framework is able to generalize to different feature extraction algorithms such as SuperPoint and feature tracking approaches.

![Overview Figure](https://dominikmuhle.github.io/dnls_covs/assets/architecture.png)

# 🏗️️ Setup

### 🐍 Python Environment

We use **Conda** to manage our Python environment:
```shell
conda env create -f environment.yml
```
Then, activate the conda environment :
```shell
conda activate dnls_covs
```

### 🪄 SuperGlue
Clone the [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) repository into the ```./scripts/thirdparty/SuperGluePretrainedNetwork/``` directory.

### 🎛️ Configuration
For configuration, we use [Hydra](https://hydra.cc) with configurations split up into small config files for easy use in ```./config/```. 

### 🪢 Pybind 
We use both pybind of [opengv](https://laurentkneip.github.io/opengv/index.html) and the [PNEC](https://github.com/tum-vision/pnec). Please follow the instruction on how to generate the pybind modules there. Place the pybind files into ```./scripts/```.

## ! As the pybind module of the PNEC is not public yet, you need to disable the import of the pypnec in the evaluation and switch the optimization framework to theseus. For the training, only the creation of the KLT-Tracks is affected. We are working on releasing the PNEC update in the next few weeks.

### 💾 Datasets
For our training we used the [KITTI visual odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset and the [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) dataset. We prepocessed the EuRoC dataset to be in the [tandem format](https://github.com/tum-vision/tandem). If you want to use any other dataset, you need add a corresponding Hydra config file.


# 🏋 Training

The config file provide the training details for our supervised and self-supervised training for the different keypoint extractors. We ran our trainings on two RTX 5000 GPUs with 16GB of memory. However, decreasing the batch size allows for training on a single GPU.

You can start the training with
```bash
python scripts/real_world.py dataset={dataset name} dataset/sweep={supervised|self_supervised} hyperparameter.batch_size={batch_size}
```

# 📊 Evaluation

You can start the evaluation with
```bash
python scripts/evaluation.py model.date={date of the stored model} model.checkpoint={checkpoint name}
```


# Citation
If you find our work useful, please consider citing our paper:
```
@article{muhle2023dnls_covs,
      title={Learning Correspondence Uncertainty via Differentiable Nonlinear Least Squares}, 
      author={Dominik Muhle and Lukas Koestler and Krishna Murthy Jatavallabhula and Daniel Cremers},
      journal={arXiv preprint arXiv:2305.09527},
      year={2023},
}
```

# Acknowledgements

This work was supported by the ERC Advanced Grant SIMULACRON, by the Munich Center for Machine Learning and by the EPSRC Programme Grant VisualAI EP/T028572/1.