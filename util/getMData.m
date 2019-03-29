function data = getMData (n)%n=1
config;
size = nnconfig.ImageSize; % 256*256
ND = nnconfig.DataNmber; %  100 
data.train = single(zeros(size));
data.label = single (zeros(size));
dir = '.\data\BRAIN_TRAIN_40LYL\';
load (strcat(dir , saveName(n,floor(log10(ND)))));% 导入路径下面的第一组图
