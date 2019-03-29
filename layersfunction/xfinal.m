function [O, DzDw ] = xfinal(xt, Y, Rho, mask, DzDy)
%% Reconstruction layer in the final of the network.
%% network setting
config;
[m ,n] = size(Y);
mask = logical( mask);
Denom1 = zeros(m ,n) ; Denom1(mask) = 1 ;
Denom2 = Rho;
Denom = Denom1+Denom2;
Denom(find(Denom == 0)) = 1e-6;
Q2=1./Denom;
%% The forward propagation
if nargin == 4
    Pr = Rho*xt;
    O = real( ifft2(( fft2 ( Pr ) + Y ) .* Q2)); % ¹«Ê½£¨6£©
end
%% The backward propagation
if nargin == 5
    %  O dE/dT
    temp1 = Q2.*fft2(DzDy);
    O = real(Rho*ifft2(temp1));
    % DzDw1
    A = (-1) * Q2 .*Q2 ;    tp1 = Q2.*fft2(xt);
    tp2 = A.*(Y + fft2(Rho*xt));
    temp2 = real(DzDy.*ifft2(tp1 + tp2));
    DzDw = sum(sum(temp2)); % dE/dRho
end
end

