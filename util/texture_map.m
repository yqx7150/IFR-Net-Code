function [ ssim_map, dc1, du] = texture_map(img1, C1)
config;
C1 = abs(C1);
img1 = double(img1);
sigma = trainOpts.sigma; 
F = trainOpts.Fold;
ksize = bitor(round(F*sigma),1);
blur_mask = fspecial('gaussian',ksize,sigma);
img2 = filter2(blur_mask, img1, 'same');
window = ones(5,5);
window = window/sum(sum(window));
mu1   = filter2(window, img1, 'same');
mu2   = filter2(window, img2, 'same');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'same') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'same') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'same') - mu1_mu2;
top = 2*sigma12 + C1; 
bottom = sigma1_sq + sigma2_sq + C1;
ssim_map1 = top./bottom;
%% the farward propagation
if nargout == 1
    ssim_map = 1-abs(ssim_map1);
else
%% the backward propagation
    ssim_map = 1-abs(ssim_map1);
    %% dE/dc1 dE/du dE/dkernel
    tp1 = filter2(rot90(window), mu1, 'same');
    tp2 = img2 + filter2(rot90(blur_mask), img1, 'same');
    dut = filter2(rot90(window), tp2, 'same')- (filter2(rot90(window), mu2, 'same') + filter2(rot90(blur_mask), tp1, 'same'));
    dut = 2*dut;
    dub1 = 2*filter2(rot90(window), img1, 'same') - 2*filter2(rot90(window), mu1, 'same');
    tp3 = filter2(rot90(blur_mask), img2, 'same');
    tp4 = filter2(rot90(window), mu2, 'same');
    dub2 = 2*filter2(rot90(window), tp3, 'same') - 2*filter2(rot90(blur_mask), tp4, 'same');
    dub = dub1 + dub2;
    ff = real(top./abs(top));
    dc1 = ff.*(top./(bottom.*bottom) - 1./bottom);
    du = (top./abs(top)).*((dub.*top)./(bottom.*bottom) - dut./bottom);
end

