function [ O,DzDw ] = zmid(p, c1, q, DzDy)
%% Nonlinear transform layer
%% z_{l} = S_PLF(c_{l}+\beta_{l})
%% The parameters are q related to the the predefined positions p;
%% Copyright (c) 2017 Yan Yang
%% All rights reserved.
%% network setting
config;
%% The forward propagation
if nargin == 3
    I = c1;
    temp = double(I);
    q = double(q);
    O = nnlinemex( p, q , temp);
end
%% The backward propagation
if nargin == 4
    I = c1;
    xvar = double(I);
    yvar = double(DzDy);
    q = double(q);
    [O, DzDw] = nnlinemex(p, q, xvar, yvar); % O is dz/d(beta)£»DzDw is dE/dq
end


