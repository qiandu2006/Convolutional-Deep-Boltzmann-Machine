%do max pool sampling in current state
%layerState -- ��ǰ���״̬
%ǰ����������ǰ���p���h���Ѿ����
function layerState = maxPooling(layerStateProb)
    'begin maxPooling'
    tic
    scale = layerStateProb.scale;
    layerState = layerStateProb;
    for n = 1 : layerStateProb.outputmaps
        for k = 1 : size(layerStateProb.p{n},1)
            for l = 1 : size(layerStateProb.p{n},2)
                for m = 1 : size(layerStateProb.p{n},3)
                    layerState.h{n}((k-1)*scale+1:k*scale,(l-1)*scale+1:l*scale,m) = 0; %״̬��գ�����ʹ��layerStateProb�����ж�
                    layerState.p{n}(k,l,m) = 0;
                    pivot = rand();
                    [x y] = maxInMatrix(layerStateProb.h{n}((k-1)*scale+1:k*scale,(l-1)*scale+1:l*scale,m));
                    if layerStateProb.h{n}((k-1)*scale+x,(l-1)*scale+y,m) > pivot
                        layerState.h{n}((k-1)*scale+x,(l-1)*scale+y,m) = 1;
                        layerState.p{n}(k,l,m) = 1;
                    end
                end
            end
        end
    end
    toc
end