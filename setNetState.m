function net = setNetState(net, netState)
    net.layers{1}.p = netState{1}.p;
    last = numel(net.layers);
    for i = 2 : last-1
        net.layers{i}.h = netState{i}.h;
        net.layers{i}.p = netState{i}.p;
    end
    net.layers{last}.label = netState{last}.label;
end