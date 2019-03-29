function [net, state, statein, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'state', 'statein','stats') ;
net = vl_simplenn_tidy(net) ;
if isempty(whos('stats'))
    error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
end
end