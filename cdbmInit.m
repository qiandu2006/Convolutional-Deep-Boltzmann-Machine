function net = cdbmInit(net, data, opts)
    'begin cdbmInit'
    %the visible layer biase
    net.layers{1}.biase = 0.01;
    net.layers{1}.p{1} = zeros([size(data{1}) opts.batchsize]);
    
    %the intermediate layers weights and biase
    for i = 2 : numel(net.layers)-1
        for k = 1 : net.layers{i}.outputmaps
            net.layers{i}.h{k} = zeros(size(net.layers{i-1}.p{1}) - [net.layers{i}.kernelsize-1 net.layers{i}.kernelsize-1 0]);
            net.layers{i}.p{k} = zeros(size(net.layers{i}.h{k}) ./ [net.layers{i}.scale net.layers{i}.scale 1]);
            for j = 1 : net.layers{i-1}.outputmaps
                net.layers{i}.w{k}{j} = 0.001 * randn([net.layers{i}.kernelsize net.layers{i}.kernelsize]);
            end
            net.layers{i}.biase(k) = 0;     %不同的outputmaps拥有不同的biase
        end
    end
    
    %the fully connnected layer
    last = numel(net.layers);
    for i = 1 : net.layers{last-1}.outputmaps
        net.layers{last}.w{i} = 0.01 * randn([net.layers{last}.classnum  size(net.layers{last-1}.p{1},1) * size(net.layers{last-1}.p{1},2)]);
    end
    net.layers{last}.label = zeros(net.layers{last}.classnum , size(net.layers{last-1}.p{1},3)); %  labels * numcases
    net.layers{last}.biase = 0;
end