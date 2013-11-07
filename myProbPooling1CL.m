function [h0,p0] = myProbPooling1CL(z)
    ocl = opencl();
    ocl.initialize(1, 1);
    ocl.addfile('myProbPooling1CL.cl');
    ocl.build();
    [d1,d2,d3] = size(z);
    d1d = int32(d1/2);
    d2d = int32(d2/2);
    d3d = int32(d3);
    z0 = clbuffer('ro', 'single', d1*d2*d3);
    h = clbuffer('rw', 'single', d1*d2*d3);
    p = clbuffer('rw', 'single', d1d*d2d*d3d);

    z0.set(z);
    h.set(zeros([d1,d2,d3]));
    p.set(zeros([d1d,d2d,d3d]));

    poolkernel = clkernel('pooling', [d1d, d2d, d3d], [d1d, d2d, d3d]);
    poolkernel(z0, h, p);

    h0 = h.get();
    p0 = p.get();
    h0 = reshape(h0,[d1,d2,d3]);
    p0 = reshape(p0,[d1d,d2d,d3d]);
end