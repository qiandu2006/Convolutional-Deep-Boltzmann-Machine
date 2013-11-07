function netRec = cdbmGradient(netRec, netMF, opts)
    last = numel(netRec.layers);
    
    netRec.layers{1}.biase = opts.momentum * netRec.layers{1}.biase + opts.alpha * mean(mean(mean(netRec.layers{1}.p{1}-netMF.layers{1}.p{1},3),2),1);%(netRec.layers{1}.biase - netMF.layers{1}.biase) / opts.batchsize;
    
    %update all hidden layers' parameters
    for i = 2 : 1 : last-1
        for j = 1 : netRec.layers{i}.outputmaps
            for k = 1 : netRec.layers{i-1}.outputmaps
                netRec.layers{i}.w{j}{k} = opts.momentum * netRec.layers{i}.w{j}{k} + opts.alpha * (convn(netRec.layers{i-1}.p{k},netRec.layers{i}.h{j},'valid') - convn(netMF.layers{i-1}.p{k},netMF.layers{i}.h{j},'valid')) / opts.batchsize - opts.decay * netRec.layers{i}.w{j}{k};
            end
            netRec.layers{i}.biase(j) = opts.momentum * netRec.layers{i}.biase(j) + opts.alpha * mean(mean(mean(netRec.layers{i}.h{k}-netMF.layers{i}.h{k},3),2),1) + opts.lambda * ( mean(mean(mean(netRec.layers{i}.h{k}+netMF.layers{i}.h{k},3),2),1) / 2 - opts.sparsity);
        end
    end
    
    %update label layer's parameters
    for i = 1 : netRec.layers{last-1}.outputmaps
        %netRec.layers{last}.w{i} = opts.momentum * netRec.layers{i}.w{i} + opts.alpha * (netRec.layers{last}.label*reshape(netRec.layers{last-1}.p{k},[size(netRec.layers{last-1}.p{k},1)*size(netRec.layers{last-1}.p{k},2) size(netRec.layers{last-1}.p{k},3)])' - netMF.layers{last}.label*reshape(netMF.layers{last-1}.p{k},[size(netMF.layers{last-1}.p{k},1)*size(netMF.layers{last-1}.p{k},2) size(netMF.layers{last-1}.p{k},3)])');
        netRec.layers{last}.w{i} = opts.momentum * netRec.layers{last}.w{i} + opts.alpha * (netRec.layers{last}.w{i} - netRec.layers{last}.w{i}) / opts.batchsize - opts.decay * netRec.layers{last}.w{i};
    end
    
    netRec.layers{last}.biase = opts.momentum * netRec.layers{last}.biase + opts.alpha * length(find(netRec.layers{last}.label ~= netMF.layers{last}.label)) / opts.batchsize;
end