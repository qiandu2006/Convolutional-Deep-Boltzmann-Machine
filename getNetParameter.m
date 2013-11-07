function netParameter = getNetParameter(net)
    last = numel(net.layers);
    for i = 2 : last-1
        netState{i}.w = net.layers{i}.w;
    end
    netState{last}.w =  net.layers{last}.w;
end