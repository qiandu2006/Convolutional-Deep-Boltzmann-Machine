close all
clear all
clc

%%准备训练数据
i = 1;
fid = fopen('F:\test\图片样本\Scene8Train\f.txt','r');
while ~feof(fid),
    fTmp = fgets(fid);
    fContent{i,1} = deblank(fTmp);      %去掉字符串中的回车号
    i = i + 1;
end
fclose(fid);

train_xO = cell(length(fContent),1);
for i = 1:size(fContent,1),
     I = imread(['F:\test\图片样本\Scene8Train\' fContent{i,1}]);
    train_xO{i} = double(I);
    train_xO{i} = train_xO{i} / 256;
end

train_yO = zeros(8, 8 * 200);
for i = 1:8,
    for j = 1:200,
        train_yO(i, (i-1)*200+j) = 1;
    end
end

%%准备测试数据
i = 1;
fid = fopen('F:\test\图片样本\Scene8Test\f.txt','r');
while ~feof(fid),
    fTmp = fgets(fid);
    fTContent{i,1} = deblank(fTmp);      %去掉字符串中的回车号
    i = i+1;
end
fclose(fid);

test_xO = cell(length(fTContent),1);
for i = 1:size(fTContent,1),
     I = imread(['F:\test\图片样本\Scene8Test\' fTContent{i,1}]);
     test_xO{i} = double(I);
     test_xO{i} = test_xO{i} / 256;
end

test_yO = zeros(8, 8 * 80);
for i = 1:8,
    for j = 1:80,
        test_yO(i, (i-1)*80+j) = 1;
    end
end

rPX = randperm(length(train_xO), length(train_xO));
for l = 1 : length(train_xO)
    train_x{l} = train_xO{rPX(l)};
    train_y(:,l) = train_yO(:,rPX(l));
end

rPY = randperm(length(test_xO), length(test_xO));
for l = 1 : length(test_xO)
    test_x{l} = test_xO{rPY(l)};
    test_y(:, l) = test_yO(:, rPY(l));
end

save MITtrain.mat train_x train_y test_x test_y