%% network setting
global trainOpts;
trainOpts.FilterNumber = 8;
trainOpts.FilterSize = 3;
trainOpts.Stage = 7;
trainOpts.Initer = 2; % inversion block number
trainOpts.Padding = 1;
trainOpts.LinearLabel = double(-1:0.02:1); % fixed
%% training and testing setting
trainOpts.sigma = 50;
trainOpts.Fold = 1;
trainOpts.batchSize = 1 ;
trainOpts.learningRate = 0.2;
trainOpts.numSubBatches = 1 ;
trainOpts.epochSize = inf;
trainOpts.prefetch = false ;
trainOpts.numEpochs = 180 ; % training number
trainOpts.weightDecay = 0.0005 ;
trainOpts.momentum = 0.95 ;
trainOpts.randomSeed = 0 ;
trainOpts.sync = false ;
trainOpts.cudnn = true ;
trainOpts.backPropDepth = +inf ;
trainOpts.continue = false ;
trainOpts.gpus = [] ;
% trainOpts.expDir = 'data/chars-experiment' ;
trainOpts.expDir = fullfile('Train_output','net') ;
trainOpts.train = [] ;
trainOpts.val = [] ;
trainOpts.solver = [] ;  % Empty array means use the default SGD solver
trainOpts.saveSolverState = true ;
trainOpts.nesterovUpdate = false ;
trainOpts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
trainOpts.profile = false ;
trainOpts.parameterServer.method = 'mmap' ;
trainOpts.parameterServer.prefix = 'mcn' ;
trainOpts.conserveMemory = false ;
trainOpts.backPropDepth = +inf ;
trainOpts.errorFunction = 'none';%multiclass' ;
trainOpts.errorLabels = {} ;
trainOpts.plotDiagnostics = false ;
trainOpts.plotStatistics = true;
trainOpts.postEpochFn = [] ;
trainOpts.skipForward = false ; 


