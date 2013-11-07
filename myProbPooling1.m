function [h, p] = myProbPooling1(matrix)
    h = exp(matrix) / (1 + sum(sum(exp(matrix))));
    p = 1 / (1 + sum(sum(exp(matrix))));
end