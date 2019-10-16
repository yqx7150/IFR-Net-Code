function [ O, O2 ,DzDw ] = Tmid(u, x, c, DzDy )
%% Feature refinement function
%% network setting
config;
tt = x - u;
[T, dc, du ] = texture_map(x,c); 
%% The forward propagation
if nargin == 3
    O = real(u + T.*tt);
end
%% The backward propagation
if nargin == 4
    temp1 = 1 - T;
    O =  DzDy.*temp1; % dTdu
    O2 = DzDy.*T; % dTdx
    temp2 =  DzDy.*tt.*dc; % dEdC
    DzDw = sum(temp2(:));
end
end
