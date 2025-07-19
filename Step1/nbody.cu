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
 * CUDA kernel to calculate new particles velocity and position
 * @param pIn  - particles in
 * @param pOut - particles out
 * @param N    - Number of particles
 * @param dt   - Size of the time step
 */
__global__ void calculateVelocity(Particles pIn, Particles pOut, const unsigned N, float dt)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) {
    return;
  }

  const float * pPosX = pIn.posX;
  const float * pPosY = pIn.posY;
  const float * pPosZ = pIn.posZ;
  const float * pWeight = pIn.weight;

  float newVelX = 0.0f;
  float newVelY = 0.0f;
  float newVelZ = 0.0f;

  const float weight = pWeight[tid];
  float adj = dt / weight;

  for (int j = 0; j < N; ++j) {
    float dx = pPosX[j] - pPosX[tid];
    float dy = pPosY[j] - pPosY[tid];
    float dz = pPosZ[j] - pPosZ[tid];

    float r2 = dx * dx + dy * dy + dz * dz;
    float r = sqrtf(r2) + __FLT_MIN__;
    float otherWeight = pWeight[j];

    if (r > COLLISION_DISTANCE) {
      float f = adj * G * weight * otherWeight / (r2 * r) + __FLT_MIN__;
      newVelX += dx * f;
      newVelY += dy * f;
      newVelZ += dz * f;
    }
    else if (r > __FLT_MIN__) {
      float velX = pIn.velX[tid];
      float velY = pIn.velY[tid];
      float velZ = pIn.velZ[tid];
      const float otherVelX = pIn.velX[j];
      const float otherVelY = pIn.velY[j];
      const float otherVelZ = pIn.velZ[j];
      newVelX += (2.f * otherWeight * otherVelX + weight * velX - otherWeight * velX) / (weight + otherWeight) - velX;
      newVelY += (2.f * otherWeight * otherVelY + weight * velY - otherWeight * velY) / (weight + otherWeight) - velY;
      newVelZ += (2.f * otherWeight * otherVelZ + weight * velZ - otherWeight * velZ) / (weight + otherWeight) - velZ;
    }
  }

  // -------------------

  pOut.posX[tid] = pPosX[tid] + (pIn.velX[tid] + newVelX) * dt;
  pOut.posY[tid] = pPosY[tid] + (pIn.velY[tid] + newVelY) * dt;
  pOut.posZ[tid] = pPosZ[tid] + (pIn.velZ[tid] + newVelZ) * dt;
  pOut.velX[tid] = pIn.velX[tid] + newVelX;
  pOut.velY[tid] = pIn.velY[tid] + newVelY;
  pOut.velZ[tid] = pIn.velZ[tid] + newVelZ;
}// end of calculate_gravitation_velocity
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
