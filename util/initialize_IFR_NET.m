function net = initialize_IFR_NET()
%% network setting
config; 
fN = trainOpts.FilterNumber;
fS = trainOpts.FilterSize; 
WD = trainOpts.weightDecay; 
LR = trainOpts.learningRate;
LL = trainOpts.LinearLabel; 
stageN = trainOpts.Stage; 
initer = trainOpts.Initer;
s = fS*fS;
padsize = floor(fS/2); 
%% network parameters
% module X 
Rho = (1e-3) * 20;
% module Z
% layer U
st = 0.5; % step size
mu2 = Rho*st;
mu1 = 1-mu2;
% layer C1
gamma = eye(s-1,fN);
B = filter_base( ); % DCT base
H = zeros(fS, fS, fN);
for i = 1 : fN
    H(:,:,i) = reshape(B*gamma(:,i),fS,fS);
    w1(:,:,1,i) = H(:,:,i); % DCT-based initialization
end
% f1 = 0.1;
% f2 = 0.1;
% w1 = f1*randn(fS,fS,1,fN,'double'); % random initialization
b1 = zeros(1,fN,'double');
% layer H 
r = (1 / 14);
linew = zeros(length(LL) , fN , 'double'); 
for i=1:fN
    linew (: , i) = nnsoft (LL, r); 
end
% layer C2
lam = 0.25;
HT = rot90(H,2);
for i = 1 : fN
    w2(:,:,i,1) = st*lam*HT(:,:,i); % DCT-based initialization
end
% w2 = f2*randn(fS,fS,fN,1,'double'); % random initialization
b2 = zeros(1,1,'double');
% module T
c = 0.008;
%% Network structure
structure_setup;
