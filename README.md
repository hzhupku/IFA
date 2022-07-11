# Learning Implicit Feature Alignment Function for Semantic Segmentation, ECCV 2022

## Abstract

Integrating high-level context information with low-level details is of central importance in semantic segmentation. Towards this
end, most existing segmentation models apply bilinear up-sampling and
convolutions to feature maps of different scales, and then align them
at the same resolution. However, bilinear up-sampling blurs the precise
information learned in these feature maps and convolutions incur extra
computation costs. To address these issues, we propose the Implicit Feature Alignment function (IFA). Our method is inspired by the rapidly expanding topic of implicit neural representations, where coordinatebased neural networks are used to designate fields of signals. In IFA, feature vectors are viewed as representing a 2D field of information.
Given a query coordinate, nearby feature vectors with their relative coordinates are taken from the multi-level feature maps and then fed into an MLP to generate the corresponding output. As such, IFA implicitly
aligns the feature maps at different levels and is capable of producing
segmentation maps in arbitrary resolutions. We demonstrate the efficacy of IFA on multiple datasets, including Cityscapes, PASCAL Context, and ADE20K. Our method can be combined with improvement
on various architectures, and it achieves state-of-the-art computationaccuracy trade-off on common benchmarks. For more details, please refer to our ECCV paper ([arxiv](https://arxiv.org/pdf/2206.08655.pdf)). 

![image](https://github.com/hzhupku/IFA/blob/main/arch.png)

## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions. 

## Training and Evaluation
```bash
cd experiments/ifa
bash train.sh
```