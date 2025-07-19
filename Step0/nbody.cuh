/**
 * @file      nbody.cuh
 *
 * @author    Ondrej Vlcek \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xvlcek27@fit.vutbr.cz
 *
 * @brief     PCG Assignment 1
 *
 * @version   2024
 *
 * @date      04 October   2023, 09:00 (created) \n
 */

#ifndef NBODY_CUH
#define NBODY_CUH

#include <cuda_runtime.h>

#include "h5Helper.h"

/**
 * Particles data structure
 */
struct Particles
{
  float* posX;
  float* posY;
  float* posZ;

  float* velX;
  float* velY;
  float* velZ;

  float* weight;
};

/**
/* Velocities data structure (to be used as buffer for partial results)
 */
struct Velocities
{
  float* x;
  float* y;
  float* z;
};


/**
 * CUDA kernel to calculate gravitation velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateGravitationVelocity(Particles      p,
                                             Velocities     tmpVel,
                                             const unsigned N,
                                             float          dt);

/**
 * CUDA kernel to calculate collision velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateCollisionVelocity(Particles      p,
                                           Velocities     tmpVel,
                                           const unsigned N,
                                           float          dt);

/**
 * CUDA kernel to update particles
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void updateParticles(Particles      p,
                                Velocities     tmpVel,
                                const unsigned N,
                                float          dt);

/**
 * CUDA kernel to calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
__global__ void centerOfMass(Particles      p,
                             float4*        com,
                             int*           lock,
                             const unsigned N);

/**
 * CPU implementation of the Center of Mass calculation
 * @param memDesc - Memory descriptor of particle data on CPU side
 */
float4 centerOfMassRef(MemDesc& memDesc);

#endif /* NBODY_H */
