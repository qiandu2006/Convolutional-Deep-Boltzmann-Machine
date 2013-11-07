function net = cdbmBottomupRec(net)
    last = numel(net.layers);
    
    for i = 2 : 1 : last-1
        %scale = net.layers{i}.scale;
        for l = 1 : net.layers{i}.outputmaps
            z0 = zeros(size(net.layers{i}.h{l}));
            for m = 1 : net.layers{i-1}.outputmaps
                z0 = z0 + convn(net.layers{i-1}.p{m}, 2*net.layers{i}.w{l}{m},'valid');
            end
            z0 = z0 + 2 * net.layers{i}.biase(l);
            z0 = sigmoid(z0);
            [net.layers{i}.h{l}, net.layers{i}.p{l}] = myProbPooling1CL(z0);
        end
        net.layers{i} = maxPoolingCL(net.layers{i});
    end
    
    %label layer
    for l = 1 : 1 : net.layers{last-1}.outputmaps
        net.layers{last}.label = net.layers{last}.label + net.layers{last}.w{l} * reshape(net.layers{last-1}.p{l}, [size(net.layers{last-1}.p{l},1)*size(net.layers{last-1}.p{l},2) size(net.layers{last-1}.p{l},3)]);
    end
    net.layers{last}.label = net.layers{last}.label + net.layers{last}.biase;
    %!!!!!!label layer sampling
    net.layers{last}.label = net.layers{last}.label ./ expand(sum(net.layers{last}.label,1),[size(net.layers{last}.label,1),1]);
    labelCumProb = cumsum(net.layers{last}.label,1);
    pivot = rand(1, size(labelCumProb,2));
    index = 0;
    for k = 1 : size(net.layers{last}.label,2)
        index = min(find(pivot(k) <= labelCumProb(:,k)));
        net.layers{last}.label = zeros(size(net.layers{last}.label));
        net.layers{last}.label(index, k) = 1;
    end
    save BottomUpRec.mat net
end
