function [net, state, statein] = processEpoch_IFR(net, state, statein, params, mode)
%% initialize with momentum 0
if isempty(state) || isempty(state.solverState)
    for i = 1:numel(net.layers)
        if isfield(net.layers{i}, 'weights')
            state.solverState{i} = cell(1, numel(net.layers{i}.weights)) ;
            state.solverState{i}(:) = {0} ;
        end
    end
end
if isempty(statein) || isempty(statein.solverState)
    for i = 1:numel(net.layerin)
        if isfield(net.layerin{i}, 'weights')
            statein.solverState{i} = cell(1, numel(net.layerin{i}.weights)) ;
            statein.solverState{i}(:) = {0} ;
        end
    end
end
parserv = [] ;
%% profile
subset = params.(mode) ;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;
res = [] ;
error = [] ;
start = tic ;
for t = 1:params.batchSize:numel(subset)
    fprintf('%s: epoch %02d: %3d/%3d:', mode, params.epoch, ...
        fix((t-1)/params.batchSize)+1, ceil(numel(subset)/params.batchSize)) ;
    batchSize = min(params.batchSize, numel(subset) - t + 1) ;
    for s = 1:params.numSubBatches
        %% get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+params.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
        im = params.imdb.images.data(:,:,t);
        label = params.imdb.images.label(:,:,t);
        mask = params.imdb.images.mask(:,:,t);
        if params.prefetch
            if s == params.numSubBatches
                batchStart = t + (labindex-1) + params.batchSize ;
                batchEnd = min(t+2*params.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
            params.getBatch(params.imdb, nextBatch) ;
        end
        if strcmp(mode, 'train')
            dzdy = 1 ;
        else
            dzdy = [] ;
        end
        net.layers{end}.class = label ;
        [res, resin] = vl_simplenn_LYL(net, im, mask, dzdy, res) ; ...   
        %% accumulate errors
        error = sum([error, [...
            sum(double(gather(res(end).x))) ;
            reshape(params.errorFunction(params, label, res),[],1) ; ]],2) ;
    end
    %% accumulate gradient
    if strcmp(mode, 'train')
        if ~isempty(parserv), parserv.sync() ; end
        [net, resin, res, state, statein] = accumulateGradients(net, resin, res, state,statein, params, batchSize, parserv) ;
    end
    %% get statistics
    time = toc(start) + adjustTime ;
    batchTime = time - stats.time ;
    stats = extractStats(net, params, error / num) ;
    stats.num = num ;
    stats.time = time ;
    currentSpeed = batchSize / batchTime ;
    averageSpeed = (t + batchSize - 1) / time ;
    if t == 3*params.batchSize + 1
        % compensate for the first three iterations, which are outliers
        adjustTime = 4*batchTime - time ;
        stats.time = time + adjustTime ;
    end
    fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
    for f = setdiff(fieldnames(stats)', {'num', 'time'})
        f = char(f) ;
        fprintf(' %s: %.3f', f, stats.(f)) ;
    end
    fprintf('\n') ;
    %% collect diagnostic statistics
    if strcmp(mode, 'train') && params.plotDiagnostics
        switchFigure(2) ; clf ;
        diagn = [res.stats] ;
        diagnvar = horzcat(diagn.variation) ;
        diagnpow = horzcat(diagn.power) ;
        subplot(2,2,1) ; barh(diagnvar) ;
        set(gca,'TickLabelInterpreter', 'none', ...
            'YTick', 1:numel(diagnvar), ...
            'YTickLabel',horzcat(diagn.label), ...
            'YDir', 'reverse', ...
            'XScale', 'log', ...
            'XLim', [1e-5 1], ...
            'XTick', 10.^(-5:1)) ;
        grid on ; title('Variation');
        subplot(2,2,2) ; barh(sqrt(diagnpow)) ;
        set(gca,'TickLabelInterpreter', 'none', ...
            'YTick', 1:numel(diagnpow), ...
            'YTickLabel',{diagn.powerLabel}, ...
            'YDir', 'reverse', ...
            'XScale', 'log', ...
            'XLim', [1e-5 1e5], ...
            'XTick', 10.^(-5:5)) ;
        grid on ; title('Power');
        subplot(2,2,3); plot(squeeze(res(end-1).x)) ;
        drawnow ;
    end
end
%% Save back to state.
state.stats.(mode) = stats ;
if ~params.saveSolverState
    state.solverState = [] ;
else
    for i = 1:numel(state.solverState)
        for j = 1:numel(state.solverState{i})
            s = state.solverState{i}{j} ;
            if isnumeric(s)
                state.solverState{i}{j} = gather(s) ;
            elseif isstruct(s)
                state.solverState{i}{j} = structfun(@gather, s, 'UniformOutput', false) ;
            end
        end
    end
    for i = 1:numel(statein.solverState)
        for j = 1:numel(statein.solverState{i})
            s = statein.solverState{i}{j} ;
            if isnumeric(s)
                statein.solverState{i}{j} = gather(s) ;
            elseif isstruct(s)
                statein.solverState{i}{j} = structfun(@gather, s, 'UniformOutput', false) ;
            end
        end
    end
end
net = vl_simplenn_move(net, 'cpu') ;