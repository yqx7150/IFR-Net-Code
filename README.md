# IFR-Net-Code
IFR-Net: Iterative Feature Refinement Network for Compressed Sensing MRI 

## About The Code
The Code based on the method described in the following paper:   
IFR-Net: Iterative Feature Refinement Network for Compressed Sensing MRI   
Author: Yiling Liu, Qiegen Liu, Minghui Zhang, Qingxin Yang, Shanshan Wang and Dong Liang       
        IEEE Trans. Comput. Imag., vol. 6, pp. 434-446, 2020.   
Version : 4.0   
The code and the algorithm are for non-comercial use only.   
Copyright 2019, Department of Electronic Information Engineering, Nanchang University.   

## Abstract
To improve the compressive sensing MRI (CS-MRI) approaches in terms of fine structure loss under high acceleration factors, we have proposed an iterative feature refinement model (IFR-CS), equipped with fixed transforms, to restore the meaningful structures and details. Nevertheless, the proposed IFR-CS still has some limitations, such as the selection of hyper-parameters, a lengthy reconstruction time, and the fixed sparsifying transform. To alleviate these issues, we unroll the iterative feature refinement procedures in IFR-CS to a supervised model-driven network, dubbed IFR-Net. Equipped with training data pairs, both regularization parameter and the utmost feature refinement operator in IFR-CS become trainable. Additionally, inspired by the powerful representation capability of convolutional neural network (CNN), CNN-based inversion blocks are explored in the sparsity-promoting denoising module to generalize the sparsity-enforcing operator. Extensive experiments on both simulated and in vivo MR datasets have shown that the proposed network possesses a strong capability to capture image details and preserve well the structural information with fast reconstruction speed.

## Overall structure of the IFR-Net

<div align=center><img src="https://github.com/yqx7150/IFR-Net-Code/blob/master/Img/Structure.png"/></div> <br>

## Result

<div align=center><img src="https://github.com/yqx7150/IFR-Net-Code/blob/master/Img/结果图.png"/></div> <br>

Fig. 1 Real-valued reconstruction results on brain image. Sampling pattern:10% pseudo radial sampling. Left to right: Ground truth, IFR-CS, ADMM-Net, and IFR-NET. <br>
<div align=center><img src="https://github.com/yqx7150/IFR-Net-Code/blob/master/Img/结果图2.png"/></div> <br>
Fig. 2 Complex-valued reconstruction results on brain image. Sampling pattern:25% 2D random sampling. Left to right: Ground truth, IFR-CS, ADMM-Net and IFR-NET.

## The link for some of compared methods

[D5C5  https://github.com/js3611/Deep-MRI-Reconstruction](https://github.com/js3611/Deep-MRI-Reconstruction) <br>
[ADMM-Net  https://github.com/yangyan92/Deep-ADMM-Net](https://github.com/yangyan92/Deep-ADMM-Net)




## Other Related Projects
  * Variable augmented neural network for decolorization and multi-exposure fusion [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1566253517305298)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DecolorNet_FusionNet_code)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)
  
  * VST-Net: Variance-stabilizing transformation inspired network for Poisson denoising [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S1047320319301439)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VST-Net)
   
  * Iterative scheme-inspired network for impulse noise removal [<font size=5>**[Paper]**</font>](https://link.springer.com/article/10.1007/s10044-018-0762-8)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/IIN-Code)

  * A Comparative Study of CNN-based Super-resolution Methods in MRI Reconstruction and Its Beyond [<font size=5>**[Paper]**</font>](https://sciencedirect.xilesou.top/science/article/abs/pii/S0923596519302358)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DCCN)

  * Progressively distribution-based Rician noise removal for magnetic resonance imaging [<font size=5>**[Paper]**</font>](http://archive.ismrm.org/2018/0773.html)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/RicianNet)
  
  * Complex-valued MRI data from SIAT   [<font size=5>**[Data]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/test_data_31)
