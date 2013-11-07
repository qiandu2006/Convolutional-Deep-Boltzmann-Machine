%   more details see : Hugo Larochelle
%   http://www.cs.toronto.edu/~larocheh/publications_en.htm : Efficient Learning of Deep Boltzmann Machines code rbm.m rbm_l2.m

function net = cdbmPretrain(net, data, label, opts)
    'begin cdbmPretrain'
%     netR1 = load('net1Pre1.mat');
%     netR2 = load('net2Pre2.mat');
%     for i = 2 : numel(net.layers)-1
%         %before training the i layer, the previous i-1 layers should be trained well
%         net = crbmPretrain(net, i, data, label, opts);
%     end
%     net.layers{2} = netR1.cdbm.layers{2};    %第一隐含层赋值
%     net.layers{3} = netR2.cdbm.layers{3};    %第二隐含层赋值
    for i = 2: 1 : numel(net.layers)-1
        %before training the i layer, the previous i-1 layers should be trained well
        net = crbmPretrain(net, i, data, label, opts);
    end
    %save netPretrained net
end