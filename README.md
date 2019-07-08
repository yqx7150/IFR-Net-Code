# IFR-Net-Code
IFR-Net: Iterative Feature Refinement Network for Fast Compressed Sensing MRI 

## About The Code
The Code based on the method described in the following paper:   
IFR-Net: Iterative Feature Refinement Network for Compressed Sensing MRI   
Author: Yiling Liu, Qiegen Liu, Minghui Zhang, Qingxin Yang, Shanshan Wang and Dong Liang   
Date : 7/2019   
Version : 3.0   
The code and the algorithm are for non-comercial use only.   
Copyright 2019, Department of Electronic Information Engineering, Nanchang University.   

## Abstract
To improve the compressive sensing MRI (CS-MRI) approaches in terms of fine structure loss under high acceleration factors, we have proposed an iterative feature refinement model (IFR-CS), equipped with fixed transforms, to restore the meaningful structures and details. Nevertheless, the proposed IFR-CS still has some limitations, such as the selection of hyper-parameters, a lengthy reconstruction time, and the fixed sparsifying transform. To alleviate these issues, we unroll the iterative feature refinement procedures in IFR-CS to a supervised model-driven network, dubbed IFR-Net. Equipped with training data pairs, both regularization parameter and the utmost feature refinement operator in IFR-CS become trainable. Additionally, inspired by the powerful representation capability of convolutional neural network (CNN), CNN-based inversion blocks are explored in the sparsity-promoting denoising module to generalize the sparsity-enforcing operator. Extensive experiments on both simulated and in vivo MR datasets have shown that the proposed network possesses a strong capability to capture image details and preserve well the structural information with fast reconstruction speed.

## Overall structure of the IFR-Net

![](https://github.com/yqx7150/IFR-Net-Code/blob/master/Img/Structure.png)

## Result

![](<div align=center>https://github.com/yqx7150/IFR-Net-Code/blob/master/Img/结果图.png</div>) <br>
Fig. 1 Real-valued reconstruction results on brain image. Sampling pattern:10% pseudo radial sampling. Left to right: Ground truth, IFR-CS, ADMM-Net, and IFR-NET. <br>
![](https://github.com/yqx7150/IFR-Net-Code/blob/master/Img/结果图2.png) <br>
Fig. 2 Complex-valued reconstruction results on brain image. Sampling pattern:25% 2D random sampling. Left to right: Ground truth, IFR-CS, ADMM-Net and IFR-NET.
