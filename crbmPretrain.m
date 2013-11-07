%max pooling sample is rather special , which should be kept in mind!!
%对网络的一层crbm进行预训练
%net表示的是整个网络，level是crbm的层号，data是训练样本，target是样本的类别标签，opts是网络参数结构
%前置条件：网络的前level-1个crbm预训练好了
%后置条件：网络的前level个crbm预训练完成
function net = crbmPretrain(net, level, data, target, opts)
    'begin crbmPretrain'
    assert((level > 1 && level < numel(net.layers)), 'crbm layer index do not in the right range');  %网络第一层为输入层，最后一层为类别标签层，都不构成crbm
    batchnum = length(data) / opts.batchsize;

    if level < numel(net.layers)-1       %预训练前几层，最后一层的训练比较特殊，因此单独处理
        %paramters changing during pretraining
        bVisInc = zeros(net.layers{level-1}.outputmaps, 1); %当前crbm的可视层的偏差增量
        wInc = cell(net.layers{level}.outputmaps, net.layers{level-1}.outputmaps); %当前crbm的weight的偏差增量
        for k = 1 : net.layers{level-1}.outputmaps
            for l = 1: net.layers{level}.outputmaps
                wInc{l}{k} = zeros(size(net.layers{level}.w{l}{k})); %同上,应该与网络中weight保持一致
            end
        end
        bHidInc = zeros(net.layers{level}.outputmaps, 1); %当前crbm的隐含层的偏差增量
        %pretrain the first crbm
        for j = 1 : batchnum       
            sprintf('batch %d pretraining...',j)
            %每个crbm的预训练都需要重新从样本集中提取数据，然后自下向上传递
            for k = 1 : opts.batchsize  %构造一个batch
                sample(:,:,k) = data{(j-1)*opts.batchsize+k};%提取样本值
            end
            %%previous level-1 layers bottom-up train
            net.layers{1}.p{1} = sample;
            for i = 1 : opts.CDstep
                for k = 2 : 1 : level-1  %如果训练的是第一层crbm，则这个循环不执行
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
                if level == 2   %没有隐含层，直接采样
                    posVisibleLayerState.p{1} =  posVisibleLayerProb.p{1} > rand(size(posVisibleLayerProb.p{1}));
                else
                    %有隐含层, 进行概率probablistic pooling
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
                    z0 = z0 + 2 * repmat(net.layers{level}.biase(k),size(z0)); %I(h(i,j),k)自下而上进行传递时，需要进行二倍补足
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
                        zp0 = zp0 + convn(net.layers{level}.h{l}, flipdim(flipdim(net.layers{level}.w{l}{k},1),2),'full'); %自上而下传递时，需要对卷积核进行反卷，并且卷积方式为扩展
                    end
                    zp0 = zp0 + repmat(net.layers{level-1}.biase(k),size(zp0));  %I(vp(i,j),k)这里的偏差不需要进行补足
                    zp0 = sigmoid(zp0); %这里加sigmoid的作用是对传入信息进行规范化
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
                        bVisInc(k)= opts.momentum * bVisInc(k) + opts.alpha * mean(mean(mean(posVisibleLayerState.h{k}-negVisibleLayerState.h{k},3),2),1) + opts.lambda * (mean(mean(mean(negVisibleLayerState.p{k},3),2),1) - opts.sparsity);%添加稀疏项
                    end
                    %update current layer weight increment
                    for l = 1: net.layers{level}.outputmaps
                        wInc{l}{k} = opts.momentum * wInc{l}{k} + opts.alpha * (convn(posVisibleLayerState.p{k},posHiddenLayerProb.h{l},'valid') - convn(negVisibleLayerState.p{k},negHiddenLayerProb.h{l},'valid')) / opts.batchsize - opts.decay * net.layers{level}.w{l}{k};
                    end
                end
                %update current layer biase increment
                for k = 1 : net.layers{level}.outputmaps  %第二项为CD动量项
                    bHidInc(k) = opts.momentum * bHidInc(k) + opts.alpha * mean(mean(mean(posHiddenLayerProb.h{k}-negHiddenLayerProb.h{k},3),2),1) + opts.lambda * (mean(mean(mean(negHiddenLayerState.p{k},3),2),1) - opts.sparsity); %添加系数项
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
        wInc = cell(net.layers{level}.outputmaps, net.layers{level-1}.outputmaps); %当前crbm的weight的偏差增量
        for k = 1 : net.layers{level-1}.outputmaps
            for l = 1: net.layers{level}.outputmaps
                wInc{l}{k} = zeros(size(net.layers{level}.w{l}{k})); %同上,应该与网络中weight保持一致
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
                        z0 = z0 + convn(posVisibleLayerState.p{l}, net.layers{level}.w{k}{l},'valid'); %最后隐含层预训练阶段向上传递时权值无需加倍
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
                    for l = 1 : net.layers{level}.outputmaps    %这里并没有选择策略：向下传递是向上传递权值的两倍。因为有来自标签层的信息
                        zp0 = zp0 + convn(net.layers{level}.h{l}, flipdim(flipdim(net.layers{level}.w{l}{k},1),2),'full');
                    end
                    zp0 = zp0 + repmat(net.layers{level-1}.biase(k),size(zp0));  %I(v(i,j),k)  %此时当前层的前一层应该是包含隐含层和池化层，因此需要进行上下概率pooling
                    zp0 = sigmoid(zp0); %规范化pooling单元到(0,1)
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
                labelCumProb = cumsum(negLabelLayerProb.label,1);   %列方向构造累计分布函数
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
                    %由于上下信息流都是没有规范化的，因此不需要sigmoid，而直接在myProbPooling中进行规范化
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