function netState = getNetState(net)
    netState{1}.p = net.layers{1}.p;
    last = numel(net.layers);
    for i = 2 : last-1
        netState{i}.h = net.layers{i}.h;
        netState{i}.p = net.layers{i}.p;
    end
    netState{last}.label =  net.layers{last}.label;
end