%ǰ�᣺�����Ѿ�ѵ������
%���ã�����������ݸ����������
%�����Ե����Ͻ��д���
function net = cdbmBottomup(net, data)
    net.layers{1}.p{1} = data;
    net.layers{1}.p{1} = net.layers{1}.p{1} > rand(size(net.layers{1}.p{1}));   %visible layer sampling
    last = numel(net.layers);
    
    for i = 2 : 1 : last-1      % for all hidden layers
        %scale = net.layers{i}.scale;
        for l = 1 : net.layers{i}.outputmaps        % for all outputmaps in current layers
            z0 = zeros(size(net.layers{i}.h{l}));   %gathering all inputmaps
            for m = 1 : net.layers{i-1}.outputmaps
                z0 = z0 + convn(net.layers{i-1}.p{m}, net.layers{i}.w{l}{m},'valid');   %!!!weight �������ڣ��Ƕ����е�inputmaps����һ��outputmaps��weight�����Ƕ�ÿһ��inputmaps��ÿһ��inputmaps����һ��weight
            end                                                                         %�ο�R. B. Palm�Ĵ��룬���ú���
            z0 = z0 + net.layers{i}.biase(l); %2 * net.layers{i}.biase(l);��Ϊ���������Ѿ�ѵ�����ˣ��Ѿ�������Ҫ�ӱ�����
            %max pooling inference
            [net.layers{i}.h{l}, net.layers{i}.p{l}] = myProbPooling1CL(z0);
        end
        net.layers{i} = maxPoolingCL(net.layers{i});%net.layers{i} = maxPooling(net.layers{i});
    end
    
    %label layer
    for l = 1 : net.layers{last-1}.outputmaps
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
end