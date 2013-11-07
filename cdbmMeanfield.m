function net = cdbmMeanfield(net, opts)
    vUnits = net.layers{1}.p{1};
    'begin cdbmMeanfield'
    for i = 1 : opts.MFstep
        net.layers{1}.p{1} = vUnits;
        net.layers{1}.p{1} = net.layers{1}.p{1} > rand(size(net.layers{1}.p{1}));
        %intermediate layer
        for k = 2 : 1 :numel(net.layers)-2;
            for l = 1 : net.layers{k}.outputmaps
                z0 = zeros(size(net.layers{k}.h{l}));
                for m = 1 : net.layers{k-1}.outputmaps
                    z0 = z0 + convn(net.layers{k-1}.p{m}, net.layers{k}.w{l}{m},'valid');
                end
                z0 = z0 + net.layers{k}.biase(l);   %I(h(i,j),k)
                %z0 = sigmoid(z0);
                zp0 = zeros(size(net.layers{k}.p{l}));
                for m = 1 : net.layers{k+1}.outputmaps
                    zp0 = zp0 + convn(net.layers{k+1}.h{m}, flipdim(flipdim(net.layers{k+1}.w{m}{l},1),2),'full');
                end
                [net.layers{k}.h{l}, net.layers{k}.p{l}] = myProbPooling2CL(z0, zp0);
            end
            %probablistic max pooling
            net.layers{k} = maxPoolingCL(net.layers{k});%net.layers{k} = maxPooling(net.layers{k});
        end

        %last hidden layer
        last = numel(net.layers)-1;
        for l = 1 : net.layers{last}.outputmaps
            z0 = zeros(size(net.layers{last}.h{l}));
            for m = 1 : net.layers{last-1}.outputmaps
                z0 = z0 + convn(net.layers{last-1}.p{m}, net.layers{last}.w{l}{m},'valid');
            end
            z0 = z0 + net.layers{last}.biase(l);   %I(h(i,j),k)
            zp0 = zeros(size(net.layers{last}.p{l}));
            zp0 = reshape(net.layers{last+1}.w{l}' * net.layers{last+1}.label,  size(net.layers{last}.p{l}));
            [net.layers{last}.h{l}, net.layers{last}.p{l}] = myProbPooling2CL(z0, zp0);
        end
        net.layers{last} = maxPoolingCL(net.layers{last});

        %label+1 layer
        for l = 1 : net.layers{last}.outputmaps
            net.layers{last+1}.label = net.layers{last+1}.label + net.layers{last+1}.w{l} * reshape(net.layers{last}.p{l}, [size(net.layers{last}.p{l},1)*size(net.layers{last}.p{l},2) size(net.layers{last}.p{l},3)]);
        end
        net.layers{last+1}.label = net.layers{last+1}.label + net.layers{last+1}.biase;
        %!!!!!!label layer sampling
        net.layers{last+1}.label = net.layers{last+1}.label ./ expand(sum(net.layers{last+1}.label,1),[size(net.layers{last+1}.label,1),1]);
        labelCumProb = cumsum(net.layers{last+1}.label,1);
        pivot = rand(1, size(labelCumProb,2));
        index = 0;
        for k = 1 : size(net.layers{last+1}.label,2)
            index = min(find(pivot(k) <= labelCumProb(:,k)));
            net.layers{last+1}.label = zeros(size(net.layers{last+1}.label));
            net.layers{last+1}.label(index, k) = 1;
        end
    end
end