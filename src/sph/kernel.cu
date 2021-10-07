#include "../../include/sph/kernel.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

CUDA_CALLABLE_MEMBER real Kernel::fixTensileInstability(Particles *particles, int p1, int p2) {

    real hbar;
    real dx[DIM];
    real W;
    real W2;
    real dWdr;
    real dWdx[DIM];

    W = 0;
    W2 = 0;
    dWdr = 0;
    for (int d = 0; d < DIM; d++) {
        dx[d] = 0.0;
        dWdx[d] = 0;
    }
    dx[0] = particles->x[p1] - particles->x[p2];
#if DIM > 1
    dx[1] = particles->y[p1] - particles->y[p2];
#if DIM > 2
    dx[2] = particles->z[p1] - particles->z[p2];
#endif
#endif

    hbar = 0.5 * (particles->sml[p1] + particles->sml[p2]);
    // calculate kernel for r and particle_distance
    //kernel(distance, hbar);
    kernel(&W, dWdx, &dWdr, dx, hbar);
    //TODO: matmean_particle_distance
    //dx[0] = matmean_particle_distance[p_rhs.materialId[a]];
    for (int d = 1; d < DIM; d++) {
        dx[d] = 0;
    }
    kernel(&W2, dWdx, &dWdr, dx, hbar);
    
    return W/W2;

}

CUDA_CALLABLE_MEMBER void Spiky::kernel(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml) {

    real r;
    real q;

    r = 0;
    for (int d = 0; d < DIM; d++) {
        r += dx[d] * dx[d];
        dWdx[d] = 0;
    }
    r = sqrt(r);
    *dWdr = 0;
    *W = 0;
    q = r/sml;

#if DIM == 1
    printf("Error, this kernel can only be used with DIM == 2,3\n");
    assert(0);
#endif

#if DIM == 2
    if (q > 1) {
        *W = 0;
    } else if (q >= 0.0) {
        *W = 10./(M_PI * sml * sml) * (1-q) * (1-q) * (1-q);
        *dWdr = -30./(M_PI * sml * sml * sml) * (1-q) * (1-q);
    }
#elif DIM == 3
    if (q > 1) {
        *W = 0;
    } else if (q >= 0.0) {
        *W = 15./(M_PI * sml * sml * sml) * (1-q) * (1-q) * (1-q);
        *dWdr = -45/(M_PI * sml * sml * sml * sml) * (1-q) * (1-q);
    }
#endif

    for (int d = 0; d < DIM; d++) {
        dWdx[d] = *dWdr/r * dx[d];
    }

}

CUDA_CALLABLE_MEMBER void CubicSpline::kernel(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml) {

    real r;
    real q;
    real f;

    r = 0;
    for (int d = 0; d < DIM; d++) {
        r += dx[d] * dx[d];
        dWdx[d] = 0;
    }
    r = sqrt(r);
    *dWdr = 0;
    *W = 0;
    q = r/sml;

    f = 4./3. * 1./sml;
#if DIM > 1
    f = 40./(7 * M_PI) * 1./(sml * sml);
#if DIM > 2
    f = 8./M_PI * 1./(sml * sml * sml);
#endif
#endif

    if (q > 1) {
        *W = 0;
        *dWdr = 0.0;

    } else if (q > 0.5) {
        *W = 2. * f * (1.-q) * (1.-q) * (1-q);
        *dWdr = -6. * f * 1./sml * (1.-q) * (1.-q);
    } else if (q <= 0.5) {
        *W = f * (6. * q * q * q - 6. * q * q + 1.);
        *dWdr = 6. * f/sml * (3 * q * q - 2 * q);
    }
    for (int d = 0; d < DIM; d++) {
        dWdx[d] = *dWdr/r * dx[d];
    }
}
