function [row col] = maxInMatrix( inmatrix)
    [C rows] = max(inmatrix); %C保存的是各列中的最大元素，row则保存的是这些元素所在行数
    [C2 col] = max(C);          %C2保存的是这些最大元素中的最大元素，col保存的是这个元素所在的列数
    row = rows(col);
end