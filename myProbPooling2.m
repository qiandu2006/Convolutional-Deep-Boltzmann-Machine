function [h, p] = myProbPooling2(hIn, pIn, scale)
    h = exp(hIn + ones(scale) * pIn ) / (1 + sum(sum(exp(hIn + ones(scale)*pIn))));
    p = 1 / (1+sum(sum(exp(hIn + ones(scale)*pIn))));
end