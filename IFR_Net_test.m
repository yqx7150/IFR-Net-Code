%% The is a test Code based on the method described in the following paper: 
% IFR-Net: Iterative Feature Refinement Network for Compressed Sensing MRI 
% Author: Yiling Liu, Qiegen Liu, Minghui Zhang, Qingxin Yang, Shanshan Wang and Dong Liang 
% Date : 3/2019 
% Version : 2.0 
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
%% Load trained network
load('./Train_output/net/IFR-Net-radial30.mat') 
%% Load data 
load('./test_data/knee02.mat') 
M0 = abs(im2double(Img));
if (length(size(M0))>2);  M0 = rgb2gray(M0);   end
M0 = M0/max(abs(M0(:)));
%% Load mask
load mask_radial30.mat;mask = fftshift(mask);
%% Undersampling in the k-space
kspace_full = fft2(M0); % K-space
y = (double(kspace_full)) .* mask;% undersampled K-space
data.train = y; % input
data.label = M0; % label
data.mask = mask; % undersampling pattern
%% reconstrction by IFR-Net
tic;
res = vl_simplenn_LYL_test(net, data);
Time_Net_rec = toc
rec_image = res(end-1).x;
re_Loss = res(end).x
re_PSnr = psnr(abs(rec_image) , abs(data.label)) 
HFEN = norm(imfilter(abs(rec_image),fspecial('log',15,1.5)) - imfilter(data.label,fspecial('log',15,1.5)),'fro')
SSIM = cal_ssim(255*rec_image, 255*data.label, 0, 0 )
Zero_filling_rec = ifft2(y); 
figure(22);imshow(abs(rec_image),[]);

