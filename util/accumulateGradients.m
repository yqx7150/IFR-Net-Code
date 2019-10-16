function [net, resin, res, state, statein] = accumulateGradients(net, resin, res, state, statein, params, batchSize, parserv)
numGpus = numel(params.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;
%% layers
for l = numel(net.layers):-1:1
    for j = numel(res(l).dzdw):-1:1
        if ~isempty(parserv)
            tag = sprintf('l%d_%d',l,j) ;
            parDer = parserv.pull(tag) ;
        else
            parDer = res(l).dzdw{j}  ;
        end
        if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
            %% special case for learning bnorm moments
            thisLR = net.layers{l}.learningRate(j) ;
            net.layers{l}.weights{j} = vl_taccum(...
                1 - thisLR, ...
                net.layers{l}.weights{j}, ...
                thisLR / batchSize, ...
                parDer) ;
        else
            %% Standard gradient training.
            if isfield(net.layers{l}, 'weightDecay')
                thisDecay = params.weightDecay * net.layerin{l}.weightDecay ;
                thisLR = params.learningRate * net.layerin{l}.learningRate ;
            else
                thisDecay = params.weightDecay;
                thisLR = params.learningRate;
            end
            if thisLR>0 || thisDecay>0
                % Normalize gradient and incorporate weight decay.
                parDer = vl_taccum(1/batchSize, parDer, ...
                    thisDecay, net.layers{l}.weights{j}) ;
                if isempty(params.solver)
                    % Default solver is the optimised SGD.
                    % Update momentum.
                    state.solverState{l}{j} = vl_taccum(...
                        params.momentum, state.solverState{l}{j}, ...
                        -1, parDer) ;
                    % Nesterov update (aka one step ahead).
                    if params.nesterovUpdate
                        delta = params.momentum * state.solverState{l}{j} - parDer ;
                    else
                        delta = state.solverState{l}{j} ;
                    end
                    % Update parameters.
                    net.layers{l}.weights{j} = vl_taccum(...
                        1, net.layers{l}.weights{j}, ...
                        thisLR, delta) ;
                else
                    % call solver function to update weights
                    [net.layers{l}.weights{j}, state.solverState{l}{j}] = ...
                        params.solver(net.layers{l}.weights{j}, state.solverState{l}{j}, ...
                        parDer, params.solverOpts, thisLR) ;
                end
            end
        end
        %% if requested, collect some useful stats for debugging
        if params.plotDiagnostics
            variation = [] ;
            label = '' ;
            switch net.layers{l}.type
                case {'conv','convt'}
                    if isnumeric(state.solverState{l}{j})
                        variation = thisLR * mean(abs(state.solverState{l}{j}(:))) ;
                    end
                    power = mean(res(l+1).x(:).^2) ;
                    if j == 1 % fiters
                        base = mean(net.layers{l}.weights{j}(:).^2) ;
                        label = 'filters' ;
                    else % biases
                        base = sqrt(power) ;%mean(abs(res(l+1).x(:))) ;
                        label = 'biases' ;
                    end
                    variation = variation / base ;
                    label = sprintf('%s_%s', net.layers{l}.name, label) ;
            end
            res(l).stats.variation(j) = variation ;
            res(l).stats.power = power ;
            res(l).stats.powerLabel = net.layers{l}.name ;
            res(l).stats.label{j} = label ;
        end
    end
end
%% layerin
for l = numel(net.layerin):-1:1
    for j = numel(resin(l).dzdw):-1:1
        if ~isempty(parserv)
            tag = sprintf('l%d_%d',l,j) ;
            parDer = parserv.pull(tag) ;
        else
            parDer = resin(l).dzdw{j}  ;
        end
        if j == 3 && strcmp(net.layerin{l}.type, 'bnorm')
            %% special case for learning bnorm moments
            thisLR = net.layerin{l}.learningRate(j) ;
            net.layerin{l}.weights{j} = vl_taccum(...
                1 - thisLR, ...
                net.layerin{l}.weights{j}, ...
                thisLR / batchSize, ...
                parDer) ;
        else
            %% Standard gradient training.
            if isfield(net.layerin{l}, 'weightDecay')
                thisDecay = params.weightDecay * net.layerin{l}.weightDecay ;
                thisLR = params.learningRate * net.layerin{l}.learningRate ;
            else
                thisDecay = params.weightDecay;
                thisLR = params.learningRate;
            end
            if thisLR>0 || thisDecay>0
                % Normalize gradient and incorporate weight decay.
                parDer = vl_taccum(1/batchSize, parDer, ...
                    thisDecay, net.layerin{l}.weights{j}) ;
                if isempty(params.solver)
                    % Default solver is the optimised SGD.
                    % Update momentum.
                    statein.solverState{l}{j} = vl_taccum(...
                        params.momentum, statein.solverState{l}{j}, ...
                        -1, parDer) ;
                    % Nesterov update (aka one step ahead).
                    if params.nesterovUpdate
                        delta = params.momentum * statein.solverState{l}{j} - parDer ;
                    else
                        delta = statein.solverState{l}{j} ;
                    end
                    % Update parameters.
                    net.layerin{l}.weights{j} = vl_taccum(...
                        1, net.layerin{l}.weights{j}, ...
                        thisLR, delta) ;
                else
                    % call solver function to update weights
                    [net.layerin{l}.weights{j}, statein.solverState{l}{j}] = ...
                        params.solver(net.layerin{l}.weights{j}, statein.solverState{l}{j}, ...
                        parDer, params.solverOpts, thisLR) ;
                end
            end
        end
        %% if requested, collect some useful stats for debugging
        if params.plotDiagnostics
            variation = [] ;
            label = '' ;
            switch net.layerin{l}.type
                case {'conv','convt'}
                    if isnumeric(statein.solverState{l}{j})
                        variation = thisLR * mean(abs(statein.solverState{l}{j}(:))) ;
                    end
                    power = mean(res(l+1).x(:).^2) ;
                    if j == 1 % fiters
                        base = mean(net.layerin{l}.weights{j}(:).^2) ;
                        label = 'filters' ;
                    else % biases
                        base = sqrt(power) ;%mean(abs(res(l+1).x(:))) ;
                        label = 'biases' ;
                    end
                    variation = variation / base ;
                    label = sprintf('%s_%s', net.layerin{l}.name, label) ;
            end
            resin(l).stats.variation(j) = variation ;
            resin(l).stats.power = power ;
            resin(l).stats.powerLabel = net.layers{l}.name ;
            resin(l).stats.label{j} = label ;
        end
    end
end