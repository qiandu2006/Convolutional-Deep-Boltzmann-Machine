%max pooling sample is rather special , which should be kept in mind!!
%�������һ��crbm����Ԥѵ��
%net��ʾ�����������磬level��crbm�Ĳ�ţ�data��ѵ��������target������������ǩ��opts����������ṹ
%ǰ�������������ǰlevel-1��crbmԤѵ������
%���������������ǰlevel��crbmԤѵ�����
function net = crbmPretrain(net, level, data, target, opts)
    'begin crbmPretrain'
    assert((level > 1 && level < numel(net.layers)), 'crbm layer index do not in the right range');  %�����һ��Ϊ����㣬���һ��Ϊ����ǩ�㣬��������crbm
    batchnum = length(data) / opts.batchsize;

    if level < numel(net.layers)-1       %Ԥѵ��ǰ���㣬���һ���ѵ���Ƚ����⣬��˵�������
        %paramters changing during pretraining
        bVisInc = zeros(net.layers{level-1}.outputmaps, 1); %��ǰcrbm�Ŀ��Ӳ��ƫ������
        wInc = cell(net.layers{level}.outputmaps, net.layers{level-1}.outputmaps); %��ǰcrbm��weight��ƫ������
        for k = 1 : net.layers{level-1}.outputmaps
            for l = 1: net.layers{level}.outputmaps
                wInc{l}{k} = zeros(size(net.layers{level}.w{l}{k})); %ͬ��,Ӧ����������weight����һ��
            end
        end
        bHidInc = zeros(net.layers{level}.outputmaps, 1); %��ǰcrbm���������ƫ������
        %pretrain the first crbm
        for j = 1 : batchnum       
            sprintf('batch %d pretraining...',j)
            %ÿ��crbm��Ԥѵ������Ҫ���´�����������ȡ���ݣ�Ȼ���������ϴ���
            for k = 1 : opts.batchsize  %����һ��batch
                sample(:,:,k) = data{(j-1)*opts.batchsize+k};%��ȡ����ֵ
            end
            %%previous level-1 layers bottom-up train
            net.layers{1}.p{1} = sample;
            for i = 1 : opts.CDstep
                for k = 2 : 1 : level-1  %���ѵ�����ǵ�һ��crbm�������ѭ����ִ��
%                     scale = net.layers{k}.scale;
                    for l = 1 : net.layers{k}.outputmaps
                        z0 = zeros(size(net.layers{k}.h{l}));
                        for m = 1 : net.layers{k-1}.outputmaps
                            z0 = z0 + convn(net.layers{k-1}.p{m}, 2*net.layers{k}.w{l}{m},'valid');
                        end
                        z0 = z0 + 2 * repmat(net.layers{k}.biase(l),size(z0));
                        %z0 = sigmoid(z0);
                        [net.layers{k}.h{l}, net.layers{k}.p{l}] = myProbPooling1CL(z0);
                    end
                end
 
                %%parameter prepare
                posVisibleLayerProb = net.layers{level-1};
                posVisibleLayerState = posVisibleLayerProb;
                negVisibleLayerProb = posVisibleLayerProb;
                negVisibleLayerState = posVisibleLayerProb;
                for k = 1 : net.layers{level-1}.outputmaps
                    if level > 2
                        posVisibleLayerState.h{k} = zeros(size(net.layers{level-1}.h{k}));
                        negVisibleLayerProb.h{k} = zeros(size(net.layers{level-1}.h{k}));
                        negVisibleLayerState.h{k} = zeros(size(net.layers{level-1}.h{k}));
                    end
                    posVisibleLayerState.p{k} = zeros(size(net.layers{level-1}.p{k}));
                    negVisibleLayerProb.p{k} = zeros(size(net.layers{level-1}.p{k}));
                    negVisibleLayerState.p{k} = zeros(size(net.layers{level-1}.p{k}));
                end
                if level == 2   %û�������㣬ֱ�Ӳ���
                    posVisibleLayerState.p{1} =  posVisibleLayerProb.p{1} > rand(size(posVisibleLayerProb.p{1}));
                else
                    %��������, ���и���probablistic pooling
                    posVisibleLayerState = maxPoolingCL(net.layers{level-1});%posVisibleLayerState = maxPooling(net.layers{level-1});
                end
                
                posHiddenLayerProb = net.layers{level};
                posHiddenLayerState = posHiddenLayerProb;
                negHiddenLayerProb = posHiddenLayerProb;
                negHiddenLayerState = posHiddenLayerProb;
                for k = 1 : net.layers{level-1}.outputmaps
                    posHiddenLayerState.h{k} = zeros(size(net.layers{level}.h{k}));
                    negHiddenLayerProb.h{k} = zeros(size(net.layers{level}.h{k}));
                    negHiddenLayerState.h{k} = zeros(size(net.layers{level}.h{k}));
                    posHiddenLayerState.p{k} = zeros(size(net.layers{level}.p{k}));
                    negHiddenLayerProb.p{k} = zeros(size(net.layers{level}.p{k}));
                    negHiddenLayerState.p{k} = zeros(size(net.layers{level}.p{k}));
                end

                %%positive phrase
                %bottom-up                
                'bottom-up'
               for k = 1 : net.layers{level}.outputmaps
                    z0 = zeros(size(net.layers{level}.h{k}));
                    for l = 1 : net.layers{level-1}.outputmaps
                        z0 = z0 + convn(net.layers{level-1}.p{l},2 * net.layers{level}.w{k}{l},'valid'); %the first crbm' pretraining weight is doubled
                    end
                    z0 = z0 + 2 * repmat(net.layers{level}.biase(k),size(z0)); %I(h(i,j),k)���¶��Ͻ��д���ʱ����Ҫ���ж�������
                    %z0 = sigmoid(z0);
                    [posHiddenLayerProb.h{k}, posHiddenLayerProb.p{k}] = myProbPooling1CL(z0);
               end
               %save posHiddenLayerProb
               posHiddenLayerState = maxPoolingCL(posHiddenLayerProb);
               %%negative phrase
               %1.top-down
               'top-down'
               for k = 1 : net.layers{level-1}.outputmaps
                    zp0 = zeros(size(net.layers{level-1}.p{k}));
                    for l = 1 : net.layers{level}.outputmaps
                        zp0 = zp0 + convn(net.layers{level}.h{l}, flipdim(flipdim(net.layers{level}.w{l}{k},1),2),'full'); %���϶��´���ʱ����Ҫ�Ծ���˽��з������Ҿ����ʽΪ��չ
                    end
                    zp0 = zp0 + repmat(net.layers{level-1}.biase(k),size(zp0));  %I(vp(i,j),k)�����ƫ���Ҫ���в���
                    zp0 = sigmoid(zp0); %�����sigmoid�������ǶԴ�����Ϣ���й淶��
                    if level == 2
                        negVisibleLayerProb.p{k} = zp0; 
                        negVisibleLayerState.p{k} = negVisibleLayerProb.p{k} > rand(size(negVisibleLayerProb.p{k}));
                    else
                        z0 = posVisibleLayerProb.h{k};
                        [negVisibleLayerProb.h{k}, negVisibleLayerProb.p{k}] = myProbPooling2CL(z0, zp0);
                    end
               end
               if level > 2
                   negVisibleLayerState = maxPoolingCL(negVisibleLayerProb);
               end
               %2.bottom-up
               'bottom-up'
               for k = 1 : net.layers{level}.outputmaps
                   z0 = zeros(size(net.layers{level}.h{k}));
                   for l = 1 : net.layers{level-1}.outputmaps
                       z0 = z0 + convn(negVisibleLayerState.p{l}, 2 * net.layers{level}.w{k}{l},'valid'); %the first crbm' pretraining weight is doubled
                   end
                   z0 = z0 + 2 * repmat(net.layers{level}.biase(k),size(z0)); %I(h(i,j),k)
                   %z0 = sigmoid(z0);
                   [negHiddenLayerProb.h{k}, negHiddenLayerProb.p{k}] = myProbPooling1CL(z0);
                end
                negHiddenLayerState = maxPoolingCL(negHiddenLayerProb);%negHiddenLayerState = maxPooling(negHiddenLayerProb);

                %%update weights
                for k = 1 : net.layers{level-1}.outputmaps
                    %update previous layer biase increment
                    if level == 2   %input layer don't regulate by sparsity
                        bVisInc(k)= opts.momentum * bVisInc(k) + opts.alpha * mean(mean(mean(posVisibleLayerState.p{k}-negVisibleLayerState.p{k},3),2),1);
                    else            %sparsity term should be considered
                        bVisInc(k)= opts.momentum * bVisInc(k) + opts.alpha * mean(mean(mean(posVisibleLayerState.h{k}-negVisibleLayerState.h{k},3),2),1) + opts.lambda * (mean(mean(mean(negVisibleLayerState.p{k},3),2),1) - opts.sparsity);%���ϡ����
                    end
                    %update current layer weight increment
                    for l = 1: net.layers{level}.outputmaps
                        wInc{l}{k} = opts.momentum * wInc{l}{k} + opts.alpha * (convn(posVisibleLayerState.p{k},posHiddenLayerProb.h{l},'valid') - convn(negVisibleLayerState.p{k},negHiddenLayerProb.h{l},'valid')) / opts.batchsize - opts.decay * net.layers{level}.w{l}{k};
                    end
                end
                %update current layer biase increment
                for k = 1 : net.layers{level}.outputmaps  %�ڶ���ΪCD������
                    bHidInc(k) = opts.momentum * bHidInc(k) + opts.alpha * mean(mean(mean(posHiddenLayerProb.h{k}-negHiddenLayerProb.h{k},3),2),1) + opts.lambda * (mean(mean(mean(negHiddenLayerState.p{k},3),2),1) - opts.sparsity); %���ϵ����
                end
                
                %%update all biase and weights
                for k = 1 : net.layers{level-1}.outputmaps
                    net.layers{level-1}.biase(k) = net.layers{level-1}.biase(k) + bVisInc(k);
                end
                for k = 1 : net.layers{level}.outputmaps
                    net.layers{level}.biase(k) = net.layers{level}.biase(k) + bHidInc(k);
                    for l = 1: net.layers{level-1}.outputmaps
                        net.layers{level}.w{k}{l} = net.layers{level}.w{k}{l} + wInc{k}{l};
                    end
                end
            end
        end
    else
        %%
        %the last crbm, label layer should be taken into consideration, the level-1, level and the label layer should all be considered
        %parameters increments initializes during pretraining
        %parameters for visible and hidden layer
        bVisInc = zeros(net.layers{level-1}.outputmaps, 1);        
        wInc = cell(net.layers{level}.outputmaps, net.layers{level-1}.outputmaps); %��ǰcrbm��weight��ƫ������
        for k = 1 : net.layers{level-1}.outputmaps
            for l = 1: net.layers{level}.outputmaps
                wInc{l}{k} = zeros(size(net.layers{level}.w{l}{k})); %ͬ��,Ӧ����������weight����һ��
            end
        end
        bHidInc = zeros(net.layers{level}.outputmaps, 1);
        wHidLabInc = cell(net.layers{level}.outputmaps, 1);
        bLabInc = 0;
        for l = 1: net.layers{level}.outputmaps
            wHidLabInc{l} = zeros(net.layers{level+1}.classnum , size(net.layers{level}.p{1},1)* size(net.layers{level}.p{1},2));
        end
        %pretrain
        for j = 1 : batchnum       
            %forward propagate
            for k = 1 : opts.batchsize
                sample(:,:,k) = data{(j-1)*opts.batchsize+k};
                net.layers{numel(net.layers)}.label(:,k) = target(:,k);
            end
            %%previous level-1 layers bottom-up train
            net.layers{1}.p{1} = sample;
            for k = 2 : 1 : level-1
                for l = 1 : net.layers{k}.outputmaps
                    z0 = zeros(size(net.layers{k}.h{l}));
                    for m = 1 : net.layers{k-1}.outputmaps
                        z0 = z0 + convn(net.layers{k-1}.p{m}, 2 * net.layers{k}.w{l}{m},'valid');
                    end
                    z0 = z0 + 2 * net.layers{k}.biase(l); %biase is also doubled
                    [net.layers{k}.h{l}, net.layers{k}.p{l}] = myProbPooling1CL(z0);
                end
            end
            for i = 1 : opts.CDstep
                %%parameter prepare, take level-1 layer as the visible layer 
                posVisibleLayerProb = net.layers{level-1};
                posVisibleLayerState = maxPoolingCL(posVisibleLayerProb);%posVisibleLayerState = maxPooling(posVisibleLayerProb);
                negVisibleLayerProb = posVisibleLayerProb;
                negVisibleLayerState = posVisibleLayerProb;
                for k = 1 : net.layers{level-1}.outputmaps
                    negVisibleLayerProb.h{k} = zeros(size(net.layers{level-1}.h{k}));
                    negVisibleLayerState.h{k} = zeros(size(net.layers{level-1}.h{k}));
                    negVisibleLayerProb.p{k} = zeros(size(net.layers{level-1}.p{k}));
                    negVisibleLayerState.p{k} = zeros(size(net.layers{level-1}.p{k}));
                end
                posHiddenLayerProb = net.layers{level};
                posHiddenLayerState = posHiddenLayerProb;%posHiddenLayerState = maxPooling(posHiddenLayerProb);
                negHiddenLayerProb = posHiddenLayerProb;
                negHiddenLayerState = posHiddenLayerProb;
                for k = 1 : net.layers{level}.outputmaps
                    posHiddenLayerState.h{k} = zeros(size(net.layers{level}.h{k}));
                    posHiddenLayerState.p{k} = zeros(size(net.layers{level}.p{k}));
                    negHiddenLayerProb.h{k} = zeros(size(net.layers{level}.h{k}));
                    negHiddenLayerState.h{k} = zeros(size(net.layers{level}.h{k}));
                    negHiddenLayerProb.p{k} = zeros(size(net.layers{level}.p{k}));
                    negHiddenLayerState.p{k} = zeros(size(net.layers{level}.p{k}));
                end
                posLabelLayer.label = net.layers{level+1}.label;
                negLabelLayerProb.label = zeros(size(net.layers{level+1}.label));
                negLabelLayer.label = zeros(size(net.layers{level+1}.label));

                %%positive phrase
                %bottom-up
                for k = 1 : net.layers{level}.outputmaps
                    %compute I(p(i,j),k) and I(h(i,j),k)
                    z0 = zeros(size(net.layers{level}.h{k}));
                    zp0 = zeros(size(net.layers{level}.p{k}));
                    for l = 1 : net.layers{level-1}.outputmaps
                        z0 = z0 + convn(posVisibleLayerState.p{l}, net.layers{level}.w{k}{l},'valid'); %���������Ԥѵ���׶����ϴ���ʱȨֵ����ӱ�
                    end
                    z0 = z0 + repmat(net.layers{level}.biase(k),size(z0)); %I(h(i,j),k)
                    zp0 = net.layers{level+1}.w{k}' * posLabelLayer.label; %I(p(i,j),k)
                    zp0 = reshape(zp0, size(net.layers{level}.p{1}));
                    [posHiddenLayerProb.h{k}, posHiddenLayerProb.p{k}] = myProbPooling2CL(z0, zp0);
                end
                posHiddenLayerState = maxPoolingCL(posHiddenLayerProb);%posHiddenLayerState = maxPooling(posHiddenLayerProb);

                %%negative phrase
                %1.top-down
                for k = 1 : net.layers{level-1}.outputmaps
                    zp0 = zeros(size(net.layers{level-1}.p{k}));
                    for l = 1 : net.layers{level}.outputmaps    %���ﲢû��ѡ����ԣ����´��������ϴ���Ȩֵ����������Ϊ�����Ա�ǩ�����Ϣ
                        zp0 = zp0 + convn(net.layers{level}.h{l}, flipdim(flipdim(net.layers{level}.w{l}{k},1),2),'full');
                    end
                    zp0 = zp0 + repmat(net.layers{level-1}.biase(k),size(zp0));  %I(v(i,j),k)  %��ʱ��ǰ���ǰһ��Ӧ���ǰ���������ͳػ��㣬�����Ҫ�������¸���pooling
                    zp0 = sigmoid(zp0); %�淶��pooling��Ԫ��(0,1)
                    z0 = negVisibleLayerProb.h{k};
                    [negVisibleLayerProb.h{k}, negVisibleLayerProb.p{k}] = myProbPooling2CL(z0, zp0);
                end
                negVisibleLayerState = maxPoolingCL(negVisibleLayerProb);
                for k = 1 : net.layers{level}.outputmaps        %so reshape p from [NP1 NP2 n] to [NP1*NP2 n]
                    negLabelLayerProb.label = negLabelLayerProb.label + net.layers{level+1}.w{k} * reshape(posHiddenLayerState.p{k},[size(posHiddenLayerState.p{k},1)*size(posHiddenLayerState.p{k},2) size(posHiddenLayerState.p{k},3)]);
                end
                negLabelLayerProb.label = negLabelLayerProb.label + net.layers{level+1}.biase;
                negLabelLayerProb.label = sigmoid(negLabelLayerProb.label); %normalizing
                %!!!!!!label sample
                negLabelLayerProb.label = negLabelLayerProb.label ./ (ones(size(negLabelLayerProb.label,1),1) * sum(negLabelLayerProb.label,1));
                labelCumProb = cumsum(negLabelLayerProb.label,1);   %�з������ۼƷֲ�����
                pivot = rand(1, size(labelCumProb,2));
                for k = 1 : size(negLabelLayer.label,2)
                    index = min(find(labelCumProb(:,k) >= pivot(k)));
                    negLabelLayer.label(index, k) = 1;
                end
                
                %2.bottom-up
                for k = 1 : net.layers{level}.outputmaps
                    z0 = zeros(size(net.layers{level}.h{k}));
                    for l = 1 : net.layers{level-1}.outputmaps
                        z0 = z0 + convn(negVisibleLayerState.p{l}, net.layers{level}.w{k}{l},'valid'); %the first crbm' pretraining weight is doubled
                    end
                    z0 = z0 + repmat(net.layers{level}.biase(k),size(z0)); %I(h(i,j),k)
                    z0 = sigmoid(z0);
                    %����������Ϣ������û�й淶���ģ���˲���Ҫsigmoid����ֱ����myProbPooling�н��й淶��
                    zp0 = reshape(net.layers{level+1}.w{k}' * negLabelLayer.label + net.layers{level+1}.biase ,size(posHiddenLayerState.p{k})); %I(p(i,j),k)
                    zp0 = sigmoid(zp0);
                    [negHiddenLayerProb.h{k}, negHiddenLayerProb.p{k}] = myProbPooling2CL(z0, zp0);
                end
                negHiddenLayerState = maxPoolingCL(negHiddenLayerProb);%negHiddenLayerState = maxPooling(negHiddenLayerProb);

                %%acquire weights
                for k = 1 : net.layers{level-1}.outputmaps
                    %bVisInc(k)= opts.momentum * bVisInc(k) + opts.alpha * mean(mean(mean(posVisibleLayerState.p{k}-negVisibleLayerState.p{k},3),2),1);
                    bVisInc(k)= opts.momentum * bVisInc(k) + opts.alpha * mean(mean(mean(posVisibleLayerState.h{k}-negVisibleLayerState.h{k},3),2),1) + opts.lambda * (mean(mean(mean(negVisibleLayerState.p{k},3),2),1) - opts.sparsity);
                    for l = 1: net.layers{level}.outputmaps
                        wInc{l}{k} = opts.momentum * wInc{l}{k} + opts.alpha * (convn(posVisibleLayerState.p{k},posHiddenLayerProb.h{l},'valid') - convn(negVisibleLayerState.p{k},negHiddenLayerProb.h{l},'valid')) / size(negHiddenLayerProb.h{l},3)  - opts.decay * net.layers{level}.w{l}{k};
                    end
                end
                for k = 1 : net.layers{level}.outputmaps
                    bHidInc(k) = opts.momentum * bHidInc(k) + opts.alpha * mean(mean(mean(posHiddenLayerProb.h{k}-negHiddenLayerProb.h{k},3),2),1);
                end
                for k = 1 : net.layers{level+1}.outputmaps
                    wHidLabInc{k} = opts.momentum * wHidLabInc{k} + opts.alpha * (posLabelLayer.label*reshape(posHiddenLayerProb.p{k},[size(posHiddenLayerProb.p{k},1)*size(posHiddenLayerProb.p{k},2) size(posHiddenLayerProb.p{k},3)])' - negLabelLayer.label*reshape(negHiddenLayerProb.p{k},[size(negHiddenLayerProb.p{k},1)*size(negHiddenLayerProb.p{k},2) size(negHiddenLayerProb.p{k},3)])');
                end
                bLabInc = opts.momentum * bLabInc + opts.alpha * length(find(posLabelLayer.label ~= negLabelLayer.label)) / size(net.layers{level+1},1) / size(net.layers{level+1},2);
                
                %%!!!!!! update network weights
                for k = 1 : net.layers{level-1}.outputmaps
                    net.layers{level-1}.biase(k) = net.layers{level-1}.biase(k) + bVisInc(k);
                    for l = 1 : net.layers{level}.outputmaps
                        net.layers{level}.w{l}{k} = net.layers{level}.w{l}{k} + wInc{l}{k};
                    end
                end
                for k = 1 : net.layers{level}.outputmaps
                    net.layers{level}.biase(k) = net.layers{level}.biase(k) + bHidInc(k);
                    net.layers{level+1}.w{k} = net.layers{level+1}.w{k} + wHidLabInc{k};
                end
                net.layers{level+1}.biase = net.layers{level+1}.biase + bLabInc;
            end
        end
    end
    %save(['layers',num2str(level)],'net.layers{level}');
end