function [ O, DzDw ] = zorg( p, I, q, DzDy1, DzDy2 )
%% The first Nonlinear transform layer
%% z_{l} = S_PLF(c_{l})
%% The parameters are q related to the the predefined positions p;
%% Copyright (c) 2017 Yan Yang
%% All rights reserved.

%% network setting
config;
%% The forward propagation
if nargin == 3
    temp = double(I);
    q = double(q);
    O = nnlinemex(p, q , temp);
end
%% The backward propagation
if nargin ==5
    DzDy = DzDy1 + DzDy2; % dE/dz
    xvar = double(I);
    yvar = double(DzDy);
    q = double(q);
    [O, DzDw] = nnlinemex(p, q, xvar, yvar);
end
end
