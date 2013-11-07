clear
clc

%�������У�p��h��Ϊ����Ϊoutputmaps��cell���飬����ÿһ��Ԫ�ؾ�Ϊά��Ϊ[w,h,batchsize]�ľ���
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
opts.alpha = 0.02;      %Ȩֵѧϰ����
opts.momentum = 0.5;
opts.decay = 0.00001;    %Ȩֵ�˻��ٶ�
opts.sparsityV = 0.2;    %���Ӳ�ϡ���
opts.sparsity = 0.1;   %������ϡ���
opts.lambda = 0.001;    %ϡ���ϵ��

opts.particleNum = 1;

load('MITtrainNormal.mat');
cdbm = cdbmInit(cdbm, train_x, opts);
cdbm = cdbmTrain(cdbm, train_x, train_y, opts);
%save cdbmTrained cdbm
errstate = cdbmTest(cdbm, test_x, test_y, opts);
save err errstate