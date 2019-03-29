%% The is a training Code based on the method described in the following paper: 
% IFR-Net: Iterative Feature Refinement Network for Compressed Sensing MRI 
% Author: Yiling Liu, Qiegen Liu, Minghui Zhang, Qingxin Yang, Shanshan Wang and Dong Liang 
% Date : 3/2019 
% Version : 2.0 
% The code and the algorithm are for non-comercial use only. 
% Copyright 2019, Department of Electronic Information Engineering, Nanchang University. 
% 
% Input:
% imdb: Set of undersampled k-space and the corresponding undersampling pattern as well as label.
%
% Optional parameters in initialize_IFR_NET:
% Rho: The penalty parameters of layer X. default: (1e-3) * 20
% st: The stepsize of gradient descend. default: 0.5
% w1: The weight of convolutional layer C1.
% b1: The bias of convolutional layer C1. 
% w2: The weight of convolutional layer C2.
% b2: The bias of convolutional layer C2. 
% linew: The initiated shrinkage function for layer H. 
% c: The parameter of feature map. default: 0.008
%
% Outputs:
% net: The trained weight.
clear all;close all;clc;
addpath(genpath('./.'))
config;
vl_compilenn;
%% Load character dataset
load('IFR-NET-radial30-knee.mat')  ;
%% Network initialization
net = initialize_IFR_NET(); 
%% Train and evaluate the IFR-NET
tic;
net = IFR_train(net, imdb, trainOpts) ;
time = toc;
time = time/3600;
fprintf('The training time is %2.1f hours.\n', time);

