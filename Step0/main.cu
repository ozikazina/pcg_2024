/**
 * @file      main.cu
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

#include <cmath>
#include <cstdio>
#include <chrono>
#include <string>

#include "nbody.cuh"
#include "h5Helper.h"

/**
 * @brief CUDA error checking macro
 * @param call CUDA API call
 */
#define CUDA_CALL(call) \
  do { \
    const cudaError_t _error = (call); \
    if (_error != cudaSuccess) \
    { \
      std::fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(_error)); \
      std::exit(EXIT_FAILURE); \
    } \
  } while(0)

/**
 * Main rotine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  if (argc != 10)
  {
    std::printf("Usage: nbody <N> <dt> <steps> <threads/block> <write intesity> <reduction threads> <reduction threads/block> <input> <output>\n");
    std::exit(1);
  }

  // Number of particles
  const unsigned N                   = static_cast<unsigned>(std::stoul(argv[1]));
  // Length of time step
  const float    dt                  = std::stof(argv[2]);
  // Number of steps
  const unsigned steps               = static_cast<unsigned>(std::stoul(argv[3]));
  // Number of thread blocks
  const unsigned simBlockDim         = static_cast<unsigned>(std::stoul(argv[4]));
  // Write frequency
  const unsigned writeFreq           = static_cast<unsigned>(std::stoul(argv[5]));
  // number of reduction threads
  const unsigned redTotalThreadCount = static_cast<unsigned>(std::stoul(argv[6]));
  // Number of reduction threads/blocks
  const unsigned redBlockDim         = static_cast<unsigned>(std::stoul(argv[7]));

  // Size of the simulation CUDA grid - number of blocks
  const unsigned simGridDim = (N + simBlockDim - 1) / simBlockDim;
  // Size of the reduction CUDA grid - number of blocks
  const unsigned redGridDim = (redTotalThreadCount + redBlockDim - 1) / redBlockDim;

  // Log benchmark setup
  std::printf("       NBODY GPU simulation\n"
              "N:                       %u\n"
              "dt:                      %f\n"
              "steps:                   %u\n"
              "threads/block:           %u\n"
              "blocks/grid:             %u\n"
              "reduction threads/block: %u\n"
              "reduction blocks/grid:   %u\n",
              N, dt, steps, simBlockDim, simGridDim, redBlockDim, redGridDim);

  const std::size_t recordsCount = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;

  Particles hParticles{};

  cudaHostAlloc(&hParticles.posX, N * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc(&hParticles.posY, N * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc(&hParticles.posZ, N * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc(&hParticles.velX, N * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc(&hParticles.velY, N * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc(&hParticles.velZ, N * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc(&hParticles.weight, N * sizeof(float), cudaHostAllocMapped);

  MemDesc md(hParticles.posX,                 1,                          0,
             hParticles.posY,                 1,                          0,
             hParticles.posZ,                 1,                          0,
             hParticles.velX,                 1,                          0,
             hParticles.velY,                 1,                          0,
             hParticles.velZ,                 1,                          0,
             hParticles.weight,               1,                          0,
             N,
             recordsCount);

  // Initialisation of helper class and loading of input data
  H5Helper h5Helper(argv[8], argv[9], md);

  try
  {
    h5Helper.init();
    h5Helper.readParticleData();
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return EXIT_FAILURE;
  }

  Particles  dParticles{};
  Velocities dTmpVelocities{};

  CUDA_CALL(cudaMalloc(&dParticles.posX, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles.posY, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles.posZ, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles.velX, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles.velY, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles.velZ, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles.weight, N * sizeof(float)));

  CUDA_CALL(cudaMalloc(&dTmpVelocities.x, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dTmpVelocities.y, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dTmpVelocities.z, N * sizeof(float)));

  CUDA_CALL(cudaMemcpy(dParticles.posX, hParticles.posX, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles.posY, hParticles.posY, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles.posZ, hParticles.posZ, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles.velX, hParticles.velX, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles.velY, hParticles.velY, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles.velZ, hParticles.velZ, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles.weight, hParticles.weight, N * sizeof(float) , cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemset(dTmpVelocities.x, 0, N * sizeof(float)));
  CUDA_CALL(cudaMemset(dTmpVelocities.y, 0, N * sizeof(float)));
  CUDA_CALL(cudaMemset(dTmpVelocities.z, 0, N * sizeof(float)));

  // Start measurement
  const auto start = std::chrono::steady_clock::now();

  for (unsigned s = 0u; s < steps; ++s)
  {
    calculateGravitationVelocity<<<simGridDim, simBlockDim>>>(dParticles, dTmpVelocities, N, dt);
    calculateCollisionVelocity<<<simGridDim, simBlockDim>>>(dParticles, dTmpVelocities, N, dt);
    updateParticles<<<simGridDim, simBlockDim>>>(dParticles, dTmpVelocities, N, dt);
  }

  // Wait for all CUDA kernels to finish
  CUDA_CALL(cudaDeviceSynchronize());

  // End measurement
  const auto end = std::chrono::steady_clock::now();

  // Approximate simulation wall time
  const float elapsedTime = std::chrono::duration<float>(end - start).count();
  std::printf("Time: %f s\n", elapsedTime);

  CUDA_CALL(cudaMemcpy(hParticles.posX, dParticles.posX, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.posY, dParticles.posY, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.posZ, dParticles.posZ, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velX, dParticles.velX, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velY, dParticles.velY, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velZ, dParticles.velZ, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Compute reference center of mass on CPU
  const float4 refCenterOfMass = centerOfMassRef(md);

  std::printf("Reference center of mass: %f, %f, %f, %f\n",
              refCenterOfMass.x,
              refCenterOfMass.y,
              refCenterOfMass.z,
              refCenterOfMass.w);

  std::printf("Center of mass on GPU: %f, %f, %f, %f\n", 0.f, 0.f, 0.f, 0.f);

  // Writing final values to the file
  h5Helper.writeComFinal(refCenterOfMass);
  h5Helper.writeParticleDataFinal();

  CUDA_CALL(cudaFree(dParticles.posX));
  CUDA_CALL(cudaFree(dParticles.posY));
  CUDA_CALL(cudaFree(dParticles.posZ));
  CUDA_CALL(cudaFree(dParticles.velX));
  CUDA_CALL(cudaFree(dParticles.velY));
  CUDA_CALL(cudaFree(dParticles.velZ));
  CUDA_CALL(cudaFree(dParticles.weight));

  CUDA_CALL(cudaFree(dTmpVelocities.x));
  CUDA_CALL(cudaFree(dTmpVelocities.y));
  CUDA_CALL(cudaFree(dTmpVelocities.z));

  CUDA_CALL(cudaFreeHost(hParticles.posX));
  CUDA_CALL(cudaFreeHost(hParticles.posY));
  CUDA_CALL(cudaFreeHost(hParticles.posZ));
  CUDA_CALL(cudaFreeHost(hParticles.velX));
  CUDA_CALL(cudaFreeHost(hParticles.velY));
  CUDA_CALL(cudaFreeHost(hParticles.velZ));
  CUDA_CALL(cudaFreeHost(hParticles.weight));

}// end of main
//----------------------------------------------------------------------------------------------------------------------
