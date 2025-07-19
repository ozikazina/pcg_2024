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
  constexpr int SH_SIZE = 64;

  extern __shared__ float pShared[];

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  const bool shouldLoad = tid < N;
  const float posX = shouldLoad ? pIn.posX[tid] : 0.0f;
  const float posY = shouldLoad ? pIn.posY[tid] : 0.0f;
  const float posZ = shouldLoad ? pIn.posZ[tid] : 0.0f;
  const float velX = shouldLoad ? pIn.velX[tid] : 0.0f;
  const float velY = shouldLoad ? pIn.velY[tid] : 0.0f;
  const float velZ = shouldLoad ? pIn.velZ[tid] : 0.0f;
  const float weight = shouldLoad ? pIn.weight[tid] : 0.0f;
  
  float newVelX = 0.0f;
  float newVelY = 0.0f;
  float newVelZ = 0.0f;

  const float adj = shouldLoad ? dt / weight : 0.0f;

  for (int k = 0; k < (N - 1) / SH_SIZE + 1; ++k)
  {
    const int sIdx = k * SH_SIZE + threadIdx.x;
    const bool shouldLoad = sIdx < N && threadIdx.x < SH_SIZE;

    if (shouldLoad) {
      pShared[threadIdx.x + 0 * SH_SIZE] = pIn.posX[sIdx];
      pShared[threadIdx.x + 1 * SH_SIZE] = pIn.posY[sIdx];
      pShared[threadIdx.x + 2 * SH_SIZE] = pIn.posZ[sIdx];
      pShared[threadIdx.x + 3 * SH_SIZE] = pIn.velX[sIdx];
      pShared[threadIdx.x + 4 * SH_SIZE] = pIn.velY[sIdx];
      pShared[threadIdx.x + 5 * SH_SIZE] = pIn.velZ[sIdx];
      pShared[threadIdx.x + 6 * SH_SIZE] = pIn.weight[sIdx];
    }

    __syncthreads();

    for (int j = 0; j < SH_SIZE && j + k * SH_SIZE < N; ++j)
    {
      const float otherPosX = pShared[j + 0 * SH_SIZE];
      const float otherPosY = pShared[j + 1 * SH_SIZE];
      const float otherPosZ = pShared[j + 2 * SH_SIZE];
      const float otherWeight = pShared[j + 6 * SH_SIZE];

      const float dx = otherPosX - posX;
      const float dy = otherPosY - posY;
      const float dz = otherPosZ - posZ;

      const float r2 = dx * dx + dy * dy + dz * dz;
      const float r = sqrtf(r2) + __FLT_MIN__;

      if (r > COLLISION_DISTANCE) {
        const float f = adj * G * weight * otherWeight / (r2 * r) + __FLT_MIN__;
        newVelX += dx * f;
        newVelY += dy * f;
        newVelZ += dz * f;
      }
      else if (r > __FLT_MIN__) {
        const float otherVelX = pShared[j + 3 * SH_SIZE];
        const float otherVelY = pShared[j + 4 * SH_SIZE];
        const float otherVelZ = pShared[j + 5 * SH_SIZE];
        newVelX += (weight * velX - otherWeight * velX + 2.f * otherWeight * otherVelX) / (weight + otherWeight) - velX;
        newVelY += (weight * velY - otherWeight * velY + 2.f * otherWeight * otherVelY) / (weight + otherWeight) - velY;
        newVelZ += (weight * velZ - otherWeight * velZ + 2.f * otherWeight * otherVelZ) / (weight + otherWeight) - velZ;
      }
    } // for j

    __syncthreads();
  } // for k
  
  if (tid >= N)
  {
    return;
  }

  // -------------------

  pOut.posX[tid] = pIn.posX[tid] + (pIn.velX[tid] + newVelX) * dt;
  pOut.posY[tid] = pIn.posY[tid] + (pIn.velY[tid] + newVelY) * dt;
  pOut.posZ[tid] = pIn.posZ[tid] + (pIn.velZ[tid] + newVelZ) * dt;
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
  extern __shared__ float4 mem[];

  mem[threadIdx.x] = {0.f, 0.f, 0.f, 0.f};

  float *const pPosX = p.posX;
  float *const pPosY = p.posY;
  float *const pPosZ = p.posZ;
  float *const pWeight = p.weight;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < N)
  {
    const float4 particle = {pPosX[tid], pPosY[tid], pPosZ[tid], pWeight[tid]};
    float4 prev = mem[threadIdx.x];

    const float4 dif = {
        particle.x - prev.x,
        particle.y - prev.y,
        particle.z - prev.z,
        (particle.w + prev.w) > 0.f ? (particle.w / (particle.w + prev.w)) : 0.f};

    prev.x += dif.x * dif.w;
    prev.y += dif.y * dif.w;
    prev.z += dif.z * dif.w;
    prev.w += particle.w;
    mem[threadIdx.x] = prev;

    tid += blockDim.x * gridDim.x;
  }

  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadIdx.x < s)
    {
      const float4 particle = mem[threadIdx.x];
      float4 prev = mem[threadIdx.x + s];

      const float4 dif = {
          particle.x - prev.x,
          particle.y - prev.y,
          particle.z - prev.z,
          (particle.w + prev.w) > 0.f ? (particle.w / (particle.w + prev.w)) : 0.f};

      prev.x += dif.x * dif.w;
      prev.y += dif.y * dif.w;
      prev.z += dif.z * dif.w;
      prev.w += particle.w;

      mem[threadIdx.x] = prev;
    }

    __syncthreads();
  }

  if (tid % blockDim.x == 0)
  {
    while (atomicCAS(lock, 0, 1) != 0)
    {
    }

    const float4 particle = mem[0];
    float4 prev = *com;

    const float4 dif = {
        particle.x - prev.x,
        particle.y - prev.y,
        particle.z - prev.z,
        (particle.w + prev.w) > 0.f ? (particle.w / (particle.w + prev.w)) : 0.f};

    prev.x += dif.x * dif.w;
    prev.y += dif.y * dif.w;
    prev.z += dif.z * dif.w;
    prev.w += particle.w;

    *com = prev;

    *lock = 0;
  }
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
