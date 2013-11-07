clear
clc

% for i=1:2
%     data.h{i} = rand([4,4]);
%     data.p{i} = zeros([2,2]);
% end
% 
% for i=1:2
%     for j=1:size(data.p{i},1)
%         for k=1:size(data.p{i},2)
%             data1.h{i}((j-1)*2+1:(j-1)*2+2,(k-1)*2+1:(k-1)*2+2)=exp(data.h{i}((j-1)*2+1:(j-1)*2+2,(k-1)*2+1:(k-1)*2+2))/sum(sum(exp(data.h{i}((j-1)*2+1:(j-1)*2+2,(k-1)*2+1:(k-1)*2+2))));
%             data1.p{i}(j,k) = 1/sum(sum(exp(data.h{i}((j-1)*2+1:(j-1)*2+2,(k-1)*2+1:(k-1)*2+2))));
%         end
%     end
% end
% data1.scale=2;
% data1.outputmaps=2;
% net.layers{last}.label = net.layers{last}.label ./ (ones(size(net.layers{last}.label,1),1) * sum(net.layers{last}.label,1));
% labelCumProb = cumsum(net.layers{last}.label,1);
% pivot = rand(1, size(labelCumProb,2));
% index = 0;
% for k = 1 : size(net.layers{last}.label,2)
%     index = min(find(pivot(k) <= labelCumProb(:,k)));
%     net.layers{last}.label(index, k) = 1;
% end

A = rand([8,10])
A = A ./ expand(sum(A,1),[size(A,1),1])
labelCumProb = cumsum(A,1)
pivot = rand(size(labelCumProb,2),1)
index = 0;
B = zeros(size(A));
for k = 1 : size(pivot,1)
    index = min(find(pivot(k) <= labelCumProb(:,k)));
    B(index,k) = 1;
end