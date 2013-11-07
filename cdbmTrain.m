function net = cdbmTrain(net, data, target, opts)
    'begin cdbmTrain'
    batchnum = length(data) / opts.batchsize;
    last = numel(net.layers);
    particleVI = cell(opts.particleNum,1);
    particleGibbs = cell(opts.MC ,1);
    for i = 1 : opts.particleNum
        particleVI{i} = getNetState(net);
    end
    for i = 1 : opts.MC 
        particleGibbs{i} = getNetState(net);
    end
    
    %%
    %pretraining
    net = cdbmPretrain(net, data, target, opts); %net = load('netPretrained.mat');%net = cdbmPretrain(net, data, target, opts);
    %save cdbmPretrained.mat net
    
    %%
    %train
    for t = 1 : opts.epoch
        netRec = net;
        randPraticle = randperm(batchnum, opts.particleNum);
        for i = 1 : opts.particleNum       
            %forward propagate
            sample = zeros([size(data{1}) opts.batchsize]);
            class = zeros(size(net.layers{last}.label));
            for k = 1 : opts.batchsize
                sample(:,:,k) = data{(randPraticle(i)-1) * opts.batchsize+k};
                class(:,k) = target(:,(randPraticle(i)-1) * opts.batchsize+k);
            end
            netRec.layers{1}.p{1} = sample;
            netRec.layers{last}.label = class;
            
            netRec = cdbmBottomupRec(netRec);
            netMF = cdbmMeanfield(netRec, opts);
            
            %weights adjustification
            netRec = cdbmGradient(netRec, netMF, opts);
            %set miu_n = miu
            particleVI{i} = getNetState(netRec); %!!!!!!still wondering, netMF or netRec?
        end
        
        %Gibbs Sample
        for i = 1 : opts.MC
            particleGibbs{i} = cdbmGibbsSample(net, particleGibbs{i}, opts);    %Gibbs sampling is changed to mean-field sampling
        end
        
       %%
        %update parameters
        %prepare
        thetaLeft = cell(last,1);
        for j = 2 : last-1
            for k = 1 : net.layers{j}.outputmaps
                for l = 1 : net.layers{j-1}.outputmaps
                    thetaLeft{j}.w{k}{l} = zeros(net.layers{j}.kernelsize, net.layers{j}.kernelsize);
                end
            end
        end
        for j = 1 : net.layers{last-1}.outputmaps
            thetaLeft{last}.w{j} = zeros(size(net.layers{last}.label,1), size(net.layers{last-1}.p{1},1)*size(net.layers{last-1}.p{1},2));
        end
        thetaRight = thetaLeft;
        for j = 2 : last-1
            for k = net.layers{j}.outputmaps
                for l = net.layers{j-1}.outputmaps
                    %compute the average weights of all miuN
                    for bI = 1 : opts.particleNum
                        thetaLeft{j}.w{k}{l} = thetaLeft{j}.w{k}{l}  + convn(particleVI{bI}{j-1}.p{l},particleVI{bI}{j}.h{k},'valid') / opts.batchsize;
                    end
                    thetaLeft{j}.w{k}{l} = thetaLeft{j}.w{k}{l} / opts.particleNum ;
                    %compute the average weights of all sampling particles
                    for pI = 1 : opts.MC
                        thetaRight{j}.w{k}{l} = thetaRight{j}.w{k}{l} + convn(particleGibbs{pI}{j-1}.p{l},particleGibbs{pI}{j}.h{k},'valid') / opts.batchsize;
                    end
                    thetaRight{j}.w{k}{l} = thetaRight{j}.w{k}{l} / opts.MC;
                    net.layers{j}.w{k}{l} = opts.momentum * net.layers{j}.w{k}{l} - opts.decay * net.layers{j}.w{k}{l} + opts.alpha * (thetaLeft{j}.w{k}{l} - thetaLeft{j}.w{k}{l});
                end
            end
        end
        
        for j = 1 : net.layers{last-1}.outputmaps
            for bI = 1 : opts.particleNum 
                thetaLeft{last}.w{j} = thetaLeft{last}.w{j}  + particleVI{bI}{last}.label * reshape(particleVI{bI}{last-1}.p{j},[size(particleVI{bI}{last-1}.p{j},1)*size(particleVI{bI}{last-1}.p{j},2) size(particleVI{bI}{last-1}.p{j},3)])'; 
            end
            thetaLeft{last}.w{j} = thetaLeft{last}.w{j} / opts.particleNum ;
            %compute the average weights of all sampling particles
            for pI = 1 : opts.MC
                thetaRight{last}.w{j} = thetaRight{last}.w{j}  + particleGibbs{pI}{last}.label * reshape(particleGibbs{pI}{last-1}.p{j},[size(particleGibbs{pI}{last-1}.p{j},1)*size(particleGibbs{pI}{last-1}.p{j},2) size(particleGibbs{pI}{last-1}.p{j},3)])'; 
            end
            thetaRight{last}.w{j} = thetaRight{last}.w{j} / opts.MC;
            net.layers{last}.w{j} = opts.momentum * net.layers{last}.w{j} - opts.decay * net.layers{last}.w{j} + opts.alpha * (thetaLeft{last}.w{j} - thetaRight{last}.w{j});
        end
    end
end
