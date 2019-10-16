function [O, DzDw] = Uorg ( x , mu2, DzDy1, DzDy2)
%% The first united layer
%% network setting
config;
mu2 = 1;
%% The forward propagation
if nargin == 2  
    O = real(mu2*x);
end
%% The backward propagation
if nargin == 4
    DzDy = DzDy1 + DzDy2;
    %  O
    O = DzDy*mu2; % dE/dx
    % DzDw
    temp = DzDy.*x; % dE/dmu2
    DzDw = sum(temp(:));
end
end