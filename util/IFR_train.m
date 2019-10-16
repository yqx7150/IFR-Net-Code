function [net, stats] = IFR_train(net, imdb, trainOpts)
if isempty(trainOpts.train), trainOpts.train = find(imdb.images.set==1) ; end  
if isempty(trainOpts.val), trainOpts.val = find(imdb.images.set==2) ; end  
%% Initialization
net.layers{end-1}.precious = 1; 
vl_simplenn_display(net, 'batchSize', trainOpts.batchSize) ;
evaluateMode = isempty(trainOpts.train) ;
trainOpts.errorFunction = @error_none ;
stats = [] ;
params = trainOpts ;
%% Train and validate
modelPath = @(ep) fullfile(trainOpts.expDir, sprintf('IFR-net-epoch-%d.mat', ep)); % The name and saving path of output
modelFigPath = fullfile(trainOpts.expDir, 'IFR-net.pdf') ; 
state = [] ;     
statein = [] ;
for epoch = 1:trainOpts.numEpochs
    %% Train for one epoch.
    params.epoch = epoch ;
    params.learningRate = trainOpts.learningRate(min(epoch, numel(trainOpts.learningRate))) ;
    params.train = trainOpts.train(randperm(numel(trainOpts.train))) ; % shuffle
    params.train = params.train(1:min(trainOpts.epochSize, numel(trainOpts.train)));
    params.val = trainOpts.val(randperm(numel(trainOpts.val))) ;
    params.imdb = imdb ;
    [net, state, statein] = processEpoch_IFR(net, state, statein, params, 'train') ; % training procedure
    [net, state, statein] = processEpoch_IFR(net, state, statein, params, 'val') ;  % validating procedure
    if ~evaluateMode
        saveState(modelPath(epoch), net, state) ;
    end
    lastStats = state.stats ;
    stats.train(epoch) = lastStats.train ;
    stats.val(epoch) = lastStats.val ;
    clear lastStats ;
    if ~evaluateMode
        saveStats(modelPath(epoch), stats) ;
    end
    %% Plot
    if params.plotStatistics
        switchFigure(1) ; clf ;
        plots = setdiff(...
            cat(2,...
            fieldnames(stats.train)', ...
            fieldnames(stats.val)'), {'num', 'time'}) ;
        for p = plots
            p = char(p) ; values = zeros(0, epoch) ;leg = {} ;
            for f = {'train', 'val'}
                f = char(f) ;
                if isfield(stats.(f), p)
                    tmp = [stats.(f).(p)] ;
                    values(end+1,:) = tmp(1,:)' ;
                    leg{end+1} = f ;
                end
            end
            subplot(1,numel(plots),find(strcmp(p,plots))) ;  plot(1:epoch, values','o-') ; xlabel('epoch') ;  title(p);  legend(leg{:}) ;
            grid on ;
        end
        drawnow ;  print(1, modelFigPath, '-dpdf') ;
    end
end
end
