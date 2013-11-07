clear
clc

%在网络中，p和h均为长度为outputmaps的cell数组，其中每一个元素均为维度为[w,h,batchsize]的矩阵
cdbm.layers = {
    struct('outputmaps', 1) %input layer
    struct('outputmaps', 8, 'kernelsize', 5,'scale', 2) %1st layer crbm
    struct('outputmaps', 8, 'kernelsize', 3,'scale', 2) %2ed layer crbm
    struct('outputmaps', 8, 'kernelsize', 3,'scale', 2) %3rd layer crbm
    struct('outputmaps', 1, 'classnum',8) %class layer
};

opts.batchsize = 40;
opts.MC = 2;
opts.MFstep = 10;
opts.epoch = 100;
opts.CDstep = 1;
opts.alpha = 0.02;      %权值学习速率
opts.momentum = 0.5;
opts.decay = 0.00001;    %权值退化速度
opts.sparsityV = 0.2;    %可视层稀疏度
opts.sparsity = 0.1;   %隐含层稀疏度
opts.lambda = 0.001;    %稀疏度系数

opts.particleNum = 1;

load('MITtrainNormal.mat');
cdbm = cdbmInit(cdbm, train_x, opts);
cdbm = cdbmTrain(cdbm, train_x, train_y, opts);
%save cdbmTrained cdbm
errstate = cdbmTest(cdbm, test_x, test_y, opts);
save err errstate