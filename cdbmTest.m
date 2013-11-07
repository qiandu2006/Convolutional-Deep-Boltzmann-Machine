function errorrate = cdbmTest(net, data, target, opts)
    'begin cdbmTest'
    %assert(length(data)==size(target,1),'data and label do not match');
    batchnum = length(data) / opts.batchsize;
    errornum = 0;
    last = numel(net.layers);
    
    for i = 1 : batchnum
        for j = 1 : opts.batchsize
            sample(:,:,j) = data{(i-1)*opts.batchsize+j};
            result(:,j) = target(:,(i-1)*opts.batchsize+j);
        end
        net = cdbmBottomup(net, sample);
        %compute the error number
        for j = 1 : opts.batchsize
            if ~isequal(net.layers{last}.label(:,j), result(:,j))
                errornum = errornum + 1;
            end
        end
    end
    errorrate = errornum/length(data);
end