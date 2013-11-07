function layerState = maxPoolingCL(layerStateProb)
%     scale = layerStateProb.scale;
    layerState = layerStateProb;
    for n = 1 : layerStateProb.outputmaps
        [layerState.h{n}, layerState.p{n}] = maxPoolingCLPri(layerStateProb.h{n}, layerStateProb.p{n});
    end    
end

function [hp0,pp0] = maxPoolingCLPri(h, p)
    ocl = opencl();
    ocl.initialize(1, 1);
    ocl.addfile('maxPoolingCL.cl');
    ocl.build();
    [hd1,hd2,hd3] = size(h);
    [pd1,pd2,pd3] = size(p);
    
    h0 = clbuffer('ro', 'single', hd1*hd2*hd3);
    p0 = clbuffer('ro', 'single', pd1*pd2*pd3);
    rn = clbuffer('ro', 'single', pd1*pd2*pd3);
    hp = clbuffer('rw', 'single', hd1*hd2*hd3);
    pp = clbuffer('rw', 'single', pd1*pd2*pd3);

    h0.set(h);
    p0.set(p);
    rn.set(rand([pd1,pd2,pd3]));
    hp.set(zeros([hd1,hd2,hd3]));
    pp.set(zeros([pd1,pd2,pd3]));

    poolkernel = clkernel('pooling', [pd1, pd2, pd3], [pd1, pd2, pd3]);
    poolkernel(h0, p0, hp, pp, rn);

    hp0 = hp.get();
    pp0 = pp.get();
    hp0 = reshape(hp0,[hd1,hd2,hd3]);
    pp0 = reshape(pp0,[pd1,pd2,pd3]);
end