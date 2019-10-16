function stats = extractStats(net, params, errors)
% -------------------------------------------------------------------------
stats.objective = errors(1) ;
for i = 1:numel(params.errorLabels)
    stats.(params.errorLabels{i}) = errors(i+1) ;
end
end