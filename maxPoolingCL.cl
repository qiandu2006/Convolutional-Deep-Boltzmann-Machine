__kernel void pooling(__global const float *h0, __global const float *p0, __global float *hp, __global float *pp, __global const float *rndnum)
{
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);

    const int gxsize = get_global_size(0);
    const int gysize = get_global_size(1);
    const int gzsize = get_global_size(2);
    
    const int begin = iz*gxsize*2*gysize*2+(ix*2)*gysize*2+iy*2;
    
    float maxnum = h0[begin];
    int px = 0, py = 0;
    
    for (int i=0;i<2;++i)
    {
        for (int j=0;j<2;++j)
        {
            if (h0[begin+i*gysize*2+j]>maxnum)
            {
                maxnum = h0[begin+i*gysize*2+j];
                px = i;
                py = j;
            }
        }
    }
    
    if (maxnum > rndnum[iz*gxsize*gysize+ix*gysize+iy])
    {
        hp[begin+px*gysize*2+py] = 1;
        pp[iz*gxsize*gysize+ix*gysize+iy] = 1;
    }
}
