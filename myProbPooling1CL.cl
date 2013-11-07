__kernel void pooling(__global const float *z0, __global float *h, __global float *p)
{
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);

    const int gxsize = get_global_size(0);
    const int gysize = get_global_size(1);
    const int gzsize = get_global_size(2);

    const int pUL = iz*gxsize*2*gysize*2+(ix*2)*gysize*2+iy*2;
    const int pUR = iz*gxsize*2*gysize*2+(ix*2)*gysize*2+iy*2+1;
    const int pDL = iz*gxsize*2*gysize*2+(ix*2+1)*gysize*2+iy*2;
    const int pDR = iz*gxsize*2*gysize*2+(ix*2+1)*gysize*2+iy*2+1;

    float ul = exp(z0[pUL]);
    float ur = exp(z0[pUR]);
    float dl = exp(z0[pDL]);
    float dr = exp(z0[pDR]);
    float total = 1 + ul + ur + dl + dr;

    h[pUL] = ul / total;
    h[pUR] = ur / total;
    h[pDL] = dl / total;
    h[pDR] = dr / total;
    
    p[iz*gxsize*gysize+ix*gysize+iy] = 1 / total;
}