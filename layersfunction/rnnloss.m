function Y = rnnloss(X, I, DzDy ) % I为原图
%% rnnloss: calculate the NMSE of restored image and original image
%% X: reconstructed image of size m*n;
%% I: ground-truth image of size m*n;
X = double(X);
I = double(I);
B = norm(I,'fro');
%% The forward propagation
if nargin == 2
    S = X - I ;
    Y = norm(S,'fro') / B ;
%% The backward propagation
elseif nargin ==3
    S = X - I ;
    Y1 = norm(S,'fro') ;
    Y = S /(B*Y1); % dE/dx,为计算X做准备
else
    error('Input arguments number not proper.');
end
end