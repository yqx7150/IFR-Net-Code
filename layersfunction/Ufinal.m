function [O, O2, O3, DzDw1, DzDw2] = Ufinal ( u, x, c2, mu1, mu2, DzDy)
%% The united layer
%% network setting
config;
%% The forward propagation
if nargin == 5
    O = real(mu1*u + mu2*x - c2);
end
%% The backward propagation
if nargin == 6
    %  O1  dE/du
    O = DzDy*mu1;
    %  O2  dE/dx
    O2 = DzDy*mu2;
    %  O3  dE/dc2
    O3 = -DzDy;
    % DzDw1
    temp1 = DzDy.*u; % dE/dmu2
    DzDw1 = sum(temp1(:));
    % DzDw2
    temp2 = DzDy.*x; % dE/dmu1
    DzDw2 = sum(temp2(:));
end
end