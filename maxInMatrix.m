function [row col] = maxInMatrix( inmatrix)
    [C rows] = max(inmatrix); %C������Ǹ����е����Ԫ�أ�row�򱣴������ЩԪ����������
    [C2 col] = max(C);          %C2���������Щ���Ԫ���е����Ԫ�أ�col����������Ԫ�����ڵ�����
    row = rows(col);
end