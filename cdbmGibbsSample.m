function stateParticle = cdbmGibbsSample(net, stateParticle, opts)
    'begin Gibbs Sampling'
    net = setNetState(net, stateParticle);
    net = cdbmBottomupRec(net);    
    stateParticle = getNetState(net);
end