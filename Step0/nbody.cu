/**
 * @file      nbody.cu
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

#include <device_launch_parameters.h>

#include "nbody.cuh"

/* Constants */
constexpr float G                  = 6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;

/**
 * CUDA kernel to calculate gravitation velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateGravitationVelocity(Particles p, Velocities tmpVel, const unsigned N, float dt)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) {
    return;
  }

  const float* pPosX = p.posX;
  const float* pPosY = p.posY;
  const float* pPosZ = p.posZ;
  float* const pWeight = p.weight;

  float newVelX = 0.0f;
  float newVelY = 0.0f;
  float newVelZ = 0.0f;

  const float posX = pPosX[tid];
  const float posY = pPosY[tid];
  const float posZ = pPosZ[tid];
  const float weight = pWeight[tid];

  for (int j = 0; j < N; ++j) {
    const float dx = pPosX[j] - posX;
    const float dy = pPosY[j] - posY;
    const float dz = pPosZ[j] - posZ;

    const float otherWeight = pWeight[j];
    const float r2 = dx * dx + dy * dy + dz * dz;
    const float r = sqrtf(r2) + __FLT_MIN__;
    const float f = G * weight * otherWeight / (r2 * r) + __FLT_MIN__;

    if (r > COLLISION_DISTANCE) {
      newVelX += dx * f;
      newVelY += dy * f;
      newVelZ += dz * f;
    }
  }

  newVelX *= dt / weight;
  newVelY *= dt / weight;
  newVelZ *= dt / weight;

  tmpVel.x[tid] = newVelX;
  tmpVel.y[tid] = newVelY;
  tmpVel.z[tid] = newVelZ;
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate collision velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateCollisionVelocity(Particles p, Velocities tmpVel, const unsigned N, float dt)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) {
    return;
  }

  float *const pPosX = p.posX;
  float *const pPosY = p.posY;
  float *const pPosZ = p.posZ;
  float *const pVelX = p.velX;
  float *const pVelY = p.velY;
  float *const pVelZ = p.velZ;
  float *const pWeight = p.weight;

  float newVelX = 0.0f;
  float newVelY = 0.0f;
  float newVelZ = 0.0f;
  
  const float posX = pPosX[tid];
  const float posY = pPosY[tid];
  const float posZ = pPosZ[tid];
  const float velX = pVelX[tid];
  const float velY = pVelY[tid];
  const float velZ = pVelZ[tid];
  const float weight = pWeight[tid];

  for (int j = 0; j < N; ++j) {
    const float dx = pPosX[j] - posX;
    const float dy = pPosY[j] - posY;
    const float dz = pPosZ[j] - posZ;

    const float r = sqrtf(dx * dx + dy * dy + dz * dz);

    if (r < COLLISION_DISTANCE && r > 0.f) {
      const float otherVelX = pVelX[j];
      const float otherVelY = pVelY[j];
      const float otherVelZ = pVelZ[j];
      const float otherWeight = pWeight[j];
      newVelX += (weight * velX - otherWeight * velX + 2.f * otherWeight * otherVelX) / (weight + otherWeight) - velX;
      newVelY += (weight * velY - otherWeight * velY + 2.f * otherWeight * otherVelY) / (weight + otherWeight) - velY;
      newVelZ += (weight * velZ - otherWeight * velZ + 2.f * otherWeight * otherVelZ) / (weight + otherWeight) - velZ;
    }
  }

  tmpVel.x[tid] += newVelX;
  tmpVel.y[tid] += newVelY;
  tmpVel.z[tid] += newVelZ;
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void updateParticles(Particles p, Velocities tmpVel, const unsigned N, float dt)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) {
    return;
  }

  float velX = p.velX[tid];
  float velY = p.velY[tid];
  float velZ = p.velZ[tid];

  velX += tmpVel.x[tid];
  velY += tmpVel.y[tid];
  velZ += tmpVel.z[tid];

  p.velX[tid] = velX;
  p.velY[tid] = velY;
  p.velZ[tid] = velZ;

  p.posX[tid] += velX * dt;
  p.posY[tid] += velY * dt;
  p.posZ[tid] += velZ * dt;
}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
__global__ void centerOfMass(Particles p, float4* com, int* lock, const unsigned N)
{

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassRef(MemDesc& memDesc)
{
  float4 com{};

  for (std::size_t i{}; i < memDesc.getDataSize(); i++)
  {
    const float3 pos = {memDesc.getPosX(i), memDesc.getPosY(i), memDesc.getPosZ(i)};
    const float  w   = memDesc.getWeight(i);

    const float4 d = {pos.x - com.x,
                      pos.y - com.y,
                      pos.z - com.z,
                      ((memDesc.getWeight(i) + com.w) > 0.0f)
                        ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w))
                        : 0.0f};

    com.x += d.x * d.w;
    com.y += d.y * d.w;
    com.z += d.z * d.w;
    com.w += w;
  }

  return com;
}// enf of centerOfMassRef
//----------------------------------------------------------------------------------------------------------------------
