function [res, resin] = vl_simplenn_LYL(net, x, mask, dzdy, res, varargin)
%VL_SIMPLENN_LYL  is derived from vl_simplenn function.
config;
opts.conserveMemory = false ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.backPropDepth = +inf ;
opts.skipForward = false ;
opts.parameterServer = [] ;
opts.holdOn = false ;
opts = vl_argparse(opts, varargin);
n = numel(net.layers);
NN = numel(net.layerin);
Ni = trainOpts.Initer - 1;
LL = trainOpts.LinearLabel;
assert(opts.backPropDepth > 0, 'Invalid `backPropDepth` value (!>0)');
backPropLim = max(n - opts.backPropDepth + 1, 1);
if (nargin <= 2) || isempty(dzdy)
    doder = false ;
    if opts.skipForward
        error('simplenn:skipForwardNoBackwPass', ...
            '`skipForward` valid only when backward pass is computed.');
    end
else
    doder = true ;
end
gpuMode = isa(x, 'gpuArray') ;
if nargin <= 3 || isempty(res)
    if opts.skipForward
        error('simplenn:skipForwardEmptyRes', ...
            'RES structure must be provided for `skipForward`.');
    end
    res = struct(...
        'x', cell(1,n+1), ...
        'dzdx', cell(1,n+1), ...
        'dzdw', cell(1,n+1), ...
        'aux', cell(1,n+1), ...
        'stats', cell(1,n+1), ...
        'time', num2cell(zeros(1,n+1)), ...
        'backwardTime', num2cell(zeros(1,n+1))) ;
    resin = struct(...
        'x', cell(1,NN+1), ...
        'dzdx', cell(1,NN+1), ...
        'dzdw', cell(1,NN+1), ...
        'aux', cell(1,NN+1), ...
        'stats', cell(1,NN+1), ...
        'time', num2cell(zeros(1,NN+1)), ...
        'backwardTime', num2cell(zeros(1,NN+1))) ;
end
if ~opts.skipForward
    res(1).x = x ;
    label = net.layers{end}.class;
end
%% Forward pass
c = - 1;
for i = 1:n
    if opts.skipForward, break; end
    l = net.layers{i} ;
    res(i).time = tic ;
    switch l.type
        case 'X_org' 
            res(i+1).x = xorg (res(i).x , mask, l.weights{1});
        case 'Z' 
            c = c + 1;
            for j = c*(4*Ni+5)+1:(c+1)*(4*Ni+5)
                li = net.layerin{j};
                switch li.type
                    case 'U_org'
                        resin(j).x = Uorg(res(i).x , li.weights{1});
                    case 'C1'
                        resin(j).x = vl_nnconv(resin(j-1).x, li.weights{1}, li.weights{2}, ...
                            'pad', li.pad, ...
                            'stride', li.stride, ...
                            'dilate', li.dilate);
                    case 'H'
                        resin(j).x = zmid(LL, resin(j-1).x, li.weights{1} );
                    case 'C2'
                        resin(j).x = vl_nnconv(resin(j-1).x, li.weights{1}, li.weights{2}, ...
                            'pad', li.pad, ...
                            'stride', li.stride, ...
                            'dilate', li.dilate); ...
                    case 'U'
                    resin(j).x = Umid(resin(j-4).x, res(i).x , resin(j-1).x,li.weights{1},li.weights{2});
                    case 'U_final'
                        resin(j).x = Ufinal(resin(j-4).x, res(i).x , resin(j-1).x,li.weights{1},li.weights{2});
                end
            end
            res(i+1).x = resin(j).x;
        case 'T'
            res(i+1).x = Tmid(res(i).x , res(i-1).x ,l.weights{1});
        case 'X_mid' 
            res(i+1).x = xmid( res(i).x, res(1).x , mask, l.weights{1});
        case 'X_final'
            res(i+1).x = xfinal( res(i).x, res(1).x , mask, l.weights{1});
        case 'loss'
            res(i+1).x = rnnloss(res(i).x, label);
        case 'custom'
            res(i+1) = l.forward(l, res(i), res(i+1)) ;
        otherwise
            error('Unknown layer type ''%s''.', l.type) ;
    end
    %% optionally forget intermediate results
    needsBProp = doder && i >= backPropLim;
    forget = opts.conserveMemory && ~needsBProp ;
    if i > 1
        lp = net.layers{i-1} ;
        % forget RELU input, even for BPROP
        forget = forget && (~needsBProp || (strcmp(l.type, 'relu') && ~lp.precious)) ;
        forget = forget && ~(strcmp(lp.type, 'loss') || strcmp(lp.type, 'softmaxloss')) ;
        forget = forget && ~lp.precious ;
    end
    if forget
        res(i).x = [] ;
    end
    if gpuMode && opts.sync
        wait(gpuDevice) ;
    end
    res(i).time = toc(res(i).time) ;
end
%% Backward pass
if doder
    res(n+1).dzdx = dzdy ;
    for i =n:-1:backPropLim
        l = net.layers{i} ;
        res(i).backwardTime = tic ;
        switch l.type
            case 'X_org'
                [res(i).dzdx{1}, res(i).dzdw{1}]  = xorg (res(1).x , mask, l.weights{1}, res(i+1).dzdx{1}, res(i+2).dzdx{2});
            case 'Z'
                for j = (c+1)*(4*Ni+5):-1:c*(4*Ni+5)+1
                    li = net.layerin{j};
                    switch li.type
                        case 'U_org'
                            [resin(j).dzdx{1}, resin(j).dzdw{1}] = Uorg(res(i).x , li.weights{1}, resin(j+1).dzdx{1},resin(j+4).dzdx{1});
                            res(i).dzdx{1} = res(i).dzdx{1} + resin(j).dzdx{1};
                        case 'C1'
                            [resin(j).dzdx{1}, resin(j).dzdw{1}, resin(j).dzdw{2}] = ...
                                vl_nnconv(resin(j-1).x, li.weights{1}, li.weights{2}, resin(j+1).dzdx{1}, ...
                                'pad', li.pad, ...
                                'stride', li.stride, ...
                                'dilate', li.dilate); ...
                        case 'H'
                        [resin(j).dzdx{1}, resin(j).dzdw{1}]  = zmid( LL, resin(j-1).x, li.weights{1}, resin(j+1).dzdx{1});
                        case 'C2'
                            [resin(j).dzdx{1}, resin(j).dzdw{1}, resin(j).dzdw{2}] = ...
                                vl_nnconv(resin(j-1).x, li.weights{1}, li.weights{2}, resin(j+1).dzdx{3}, ...
                                'pad', li.pad, ...
                                'stride', li.stride, ...
                                'dilate', li.dilate);
                        case 'U'
                            [resin(j).dzdx{1},resin(j).dzdx{2},resin(j).dzdx{3}, resin(j).dzdw{1}, resin(j).dzdw{2}] = Umid(resin(j-4).x, res(i).x , resin(j-1).x,li.weights{1},li.weights{2},resin(j+1).dzdx{1},resin(j+4).dzdx{1});
                             res(i).dzdx{1} = res(i).dzdx{1} + resin(j).dzdx{2};
                        case 'U_final'
                            [resin(j).dzdx{1},resin(j).dzdx{2},resin(j).dzdx{3}, resin(j).dzdw{1}, resin(j).dzdw{2}] = Ufinal(resin(j-4).x, res(n).x , resin(j-1).x,li.weights{1},li.weights{2},res(i+1).dzdx{1});
                            res(i).dzdx{1} = resin(j).dzdx{2};
                    end
                end
                c = c - 1;
            case 'T'
                [ res(i).dzdx{1}, res(i).dzdx{2}, res(i).dzdw{1} ] = Tmid(res(i).x , res(i-1).x, l.weights{1}, res(i+1).dzdx{1});
            case 'X_mid' 
                [res(i).dzdx{1}, res(i).dzdw{1}] = xmid(res(i).x , res(1).x , mask, l.weights{1},  res(i+1).dzdx{1},res(i+2).dzdx{2});
            case 'X_final' 
                [res(i).dzdx{1}, res(i).dzdw{1}] = xfinal(res(i).x , res(1).x , mask, l.weights{1}, res(i+1).dzdx{1});
            case 'loss' 
                res(i).dzdx{1} = rnnloss(res(i).x, label, 1);
            case 'custom'
                res(i) = l.backward(l, res(i), res(i+1)) ;
        end % layers
        if opts.conserveMemory && ~net.layers{i}.precious && i ~= n
            res(i+1).dzdx = [] ;
            res(i+1).x = [] ;
        end
        if gpuMode && opts.sync
            wait(gpuDevice) ;
        end
        res(i).backwardTime = toc(res(i).backwardTime) ;
    end
    if i > 1 && i == backPropLim && opts.conserveMemory && ~net.layers{i}.precious
        res(i).dzdx = [] ;
        res(i).x = [] ;
    end
end
