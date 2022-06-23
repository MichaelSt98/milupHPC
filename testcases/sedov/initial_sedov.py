import seagen
import numpy as np
import h5py


def cubicSpline(dx_vec, sml):
    r = 0
    for d in range(3):
        r += dx_vec[d] * dx_vec[d]

    r = np.sqrt(r)
    W = 0
    q = r/sml

    f = 8./np.pi * 1./(sml * sml * sml);

    if q > 1:
        W = 0
    elif q > 0.5:
        W = 2. * f * (1.-q) * (1.-q) * (1-q)
    elif q <= 0.5:
        W = f * (6. * q * q * q - 6. * q * q + 1.)

    return W

if __name__ == '__main__':

    N = 61  # num particles = N**3

    sml = 0.041833  # for n = 61**3  = 226981  = ca. 2e5
    # sml = 0.031375  # for n = 81**3  = 531441  = ca. 5e5
    # sml = 0.0251    # for n = 101**3 = 1030301 = ca. 1e6
    # sml = 0.02      # for n = 126**3 = 2000376 = ca 2e6
    # sml = 0.01476   # for n = 171**3 = 5000211 = ca 5e6

    r_smooth = 2*sml

    h5f = h5py.File("sedov_N{}.h5".format(N), "w")
    print("Saving to sedov_N{}.h5 ...".format(N))

    N = N**3

    explosion_energy = 1.0
    efloor = 1e-6

    radii = np.arange(0.001, 0.5, 0.001)
    densities = np.ones(len(radii))     # e.g. constant density

    particles = seagen.GenSphere(N, radii, densities)

    particlesSedov = len(particles.x)
    pos = np.zeros((len(particles.x), 3))
    vel = np.zeros((len(particles.x), 3))
    u = np.zeros(len(particles.x))
    mass = np.zeros(len(particles.x))
    materialId = np.zeros(len(particles.x), dtype=np.int8)

    numParticles = 0
    verify = 0.
    for i in range(len(particles.x)):
        ri = np.sqrt(particles.x[i]**2 + particles.y[i]**2 + particles.z[i]**2)
        W = cubicSpline(np.array([particles.x[i], particles.y[i], particles.z[i]]), r_smooth)
        e = W
        verify += W * particles.m[i]
        if e < 1e-6:
            particlesSedov -= 1
            e = 1e-6
        pos[numParticles, 0] = particles.x[i]
        pos[numParticles, 1] = particles.y[i]
        pos[numParticles, 2] = particles.z[i]
        vel[numParticles, 0] = float(0)
        vel[numParticles, 1] = float(0)
        vel[numParticles, 2] = float(0)
        mass[numParticles] = particles.m[i]
        u[numParticles] = e
        numParticles += 1

    # print("verify: {}".format(verify))
    # print("min: {} | max: {}".format(min(particles.x), max(particles.x)))

    print("Writing to HDF5 file ...")

    h5f.create_dataset("x", data=pos)
    h5f.create_dataset("v", data=vel)
    h5f.create_dataset("m", data=mass)
    h5f.create_dataset("materialId", data=materialId)
    h5f.create_dataset("u", data=u)

    h5f.close()

    print("Finished!")
