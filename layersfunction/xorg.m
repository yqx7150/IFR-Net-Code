function [ O , DzDw ] = xorg( Y, Rho,mask, DzDy1, DzDy2 )
%% The first Reconstruction layer
%% network setting
config;
[m ,n] = size(Y);
%% prepare for the equation x(1)
mask = logical(mask);
Denom1 = zeros(m , n) ; Denom1(mask) = 1 ;
Denom2 = Rho;
Denom = Denom1 + Denom2;  % diagonal matrix
Denom(find( Denom  == 0)) = 1e-6;
Q1 = 1./ Denom;
%% The forward propagation
if nargin == 3
    O = real(ifft2(Y .* Q1));
end

%% The backward propagation
if nargin == 5
    DzDy = DzDy1 + DzDy2;
    %O
    O = 1;
    % DzDw1
    A = (-1) * Q1 .*Q1 ;
%     temp = DzDy.*ifft2(A.*Y);
    temp = real(DzDy.*ifft2(A.*Y));
    DzDw = sum(sum(temp)); % dE/dRho
end
end
