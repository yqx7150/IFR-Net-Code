%% The is a test Code based on the method described in the following paper: 
% IFR-Net: Iterative Feature Refinement Network for Compressed Sensing MRI 
% Author: Yiling Liu, Qiegen Liu, Minghui Zhang, Qingxin Yang, Shanshan Wang and Dong Liang 
% Date : 7/2019 
% Version : 3.0 
% The code and the algorithm are for non-comercial use only. 
% Copyright 2019, Department of Electronic Information Engineering, Nanchang University. 
% 
% Input:
% M0: The corresponding ground truth image.
% mask: The undersampling pattern.
% y: Undersampled k-space.
% net: The trained net.
%
% Outputs:
% re_Loss: The loss of denoised image.
% rec_image: The reconstructed image.
clc; clear;close all;
%% add path
addpath('./layersfunction')
addpath('./mask')
addpath('./matconvnet')
addpath('./Test_data')
addpath('./Train_data')
addpath('./Train_output')
addpath('./util')
addpath(genpath(pwd))
vl_compilenn; % prepare for MatConvNet
%% Load trained network
load('./Train_output/net/IFR-Net-radial30.mat') 
%% Load data 
load('./test_data/knee02.mat') 
M0 = abs(im2double(Img));
if (length(size(M0))>2);  M0 = rgb2gray(M0);   end
M0 = (M0 - min(M0(:)))/(max(M0(:)) - min(M0(:))); % normalization
%% Load mask
load mask_radial30.mat; mask = fftshift(mask);
%% Undersampling in the k-space
kspace_full = fft2(M0); % full K-space
y = (double(kspace_full)) .* mask;% undersampled K-space
data.train = y; % input
data.label = M0; % label
data.mask = mask; % undersampling pattern
%% reconstruction by IFR-Net
tic;
res = vl_simplenn_LYL_test(net, data); % reconstrution procedure
Time_Net_rec = toc;
rec_image = res(end-1).x;
%% evaluation
re_Loss = res(end).x;
re_PSNR = psnr(abs(rec_image) , abs(data.label)) ;
HFEN = norm(imfilter(abs(rec_image),fspecial('log',15,1.5)) - imfilter(data.label,fspecial('log',15,1.5)),'fro');
SSIM = cal_ssim(255*rec_image, 255*data.label, 0, 0 );
fprintf('IFR-Net reconstruction: NMSE %.4f, PSNR %.4f, HFEN %.4f, SSIM %.4f.\n',re_Loss, re_PSNR, HFEN, SSIM);
figure(22);imshow(abs(rec_image),[]);

