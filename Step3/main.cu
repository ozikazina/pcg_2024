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
  float4*   hCenterOfMass{};

  cudaHostAlloc(&hParticles.posX, N * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc(&hParticles.posY, N * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc(&hParticles.posZ, N * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc(&hParticles.velX, N * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc(&hParticles.velY, N * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc(&hParticles.velZ, N * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc(&hParticles.weight, N * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc(&hCenterOfMass, 4 * sizeof(float), cudaHostAllocMapped);

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

  Particles dParticles[2]{};
  float4*   dCenterOfMass{};
  int*      dLock{};

  CUDA_CALL(cudaMalloc(&dParticles[0].posX, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles[0].posY, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles[0].posZ, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles[0].velX, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles[0].velY, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles[0].velZ, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles[0].weight, N * sizeof(float)));

  CUDA_CALL(cudaMalloc(&dParticles[1].posX, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles[1].posY, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles[1].posZ, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles[1].velX, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles[1].velY, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles[1].velZ, N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dParticles[1].weight, N * sizeof(float)));

  CUDA_CALL(cudaMalloc(&dCenterOfMass, 4 * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dLock, sizeof(int)));

  CUDA_CALL(cudaMemcpy(dParticles[0].posX, hParticles.posX, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].posY, hParticles.posY, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].posZ, hParticles.posZ, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].velX, hParticles.velX, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].velY, hParticles.velY, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].velZ, hParticles.velZ, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].weight, hParticles.weight, N * sizeof(float) , cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(dParticles[1].posX, dParticles[0].posX, N * sizeof(float), cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].posY, dParticles[0].posY, N * sizeof(float), cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].posZ, dParticles[0].posZ, N * sizeof(float), cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].velX, dParticles[0].velX, N * sizeof(float), cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].velY, dParticles[0].velY, N * sizeof(float), cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].velZ, dParticles[0].velZ, N * sizeof(float), cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].weight, dParticles[0].weight, N * sizeof(float) , cudaMemcpyDeviceToDevice));

  CUDA_CALL(cudaMemset(dLock, 0, sizeof(int)));
  
  CUDA_CALL(cudaMemset(dCenterOfMass, 0, 4 * sizeof(float)));

  // Get CUDA device warp size
  int device;
  int warpSize;

  CUDA_CALL(cudaGetDevice(&device));
  CUDA_CALL(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device));

  const std::size_t sharedMemSize    = 7 * 64 * sizeof(float);
  const std::size_t redSharedMemSize = 4 * sizeof(float) * redBlockDim;   // you can use warpSize variable

  // Start measurement
  const auto start = std::chrono::steady_clock::now();

  for (unsigned s = 0u; s < steps; ++s)
  {
    const unsigned srcIdx = s % 2;        // source particles index
    const unsigned dstIdx = (s + 1) % 2;  // destination particles index

    calculateVelocity<<<simGridDim, simBlockDim, sharedMemSize>>>(dParticles[srcIdx], dParticles[dstIdx], N, dt);
  }

  const unsigned resIdx = steps % 2;    // result particles index

  centerOfMass<<<redGridDim,redBlockDim,redSharedMemSize>>>(dParticles[resIdx], dCenterOfMass, dLock, N);

  // Wait for all CUDA kernels to finish
  CUDA_CALL(cudaDeviceSynchronize());

  // End measurement
  const auto end = std::chrono::steady_clock::now();

  // Approximate simulation wall time
  const float elapsedTime = std::chrono::duration<float>(end - start).count();
  std::printf("Time: %f s\n", elapsedTime);

  CUDA_CALL(cudaMemcpy(hParticles.posX, dParticles[resIdx].posX, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.posY, dParticles[resIdx].posY, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.posZ, dParticles[resIdx].posZ, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velX, dParticles[resIdx].velX, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velY, dParticles[resIdx].velY, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velZ, dParticles[resIdx].velZ, N * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaMemcpy(hCenterOfMass, dCenterOfMass, 4 * sizeof(float), cudaMemcpyDeviceToHost));


  // Compute reference center of mass on CPU
  const float4 refCenterOfMass = centerOfMassRef(md);

  std::printf("Reference center of mass: %f, %f, %f, %f\n",
              refCenterOfMass.x,
              refCenterOfMass.y,
              refCenterOfMass.z,
              refCenterOfMass.w);

  std::printf("Center of mass on GPU: %f, %f, %f, %f\n",
              hCenterOfMass->x,
              hCenterOfMass->y,
              hCenterOfMass->z,
              hCenterOfMass->w);

  // Writing final values to the file
  h5Helper.writeComFinal(*hCenterOfMass);
  h5Helper.writeParticleDataFinal();

  CUDA_CALL(cudaFree(dParticles[0].posX));
  CUDA_CALL(cudaFree(dParticles[0].posY));
  CUDA_CALL(cudaFree(dParticles[0].posZ));
  CUDA_CALL(cudaFree(dParticles[0].velX));
  CUDA_CALL(cudaFree(dParticles[0].velY));
  CUDA_CALL(cudaFree(dParticles[0].velZ));
  CUDA_CALL(cudaFree(dParticles[0].weight));

  CUDA_CALL(cudaFree(dParticles[1].posX));
  CUDA_CALL(cudaFree(dParticles[1].posY));
  CUDA_CALL(cudaFree(dParticles[1].posZ));
  CUDA_CALL(cudaFree(dParticles[1].velX));
  CUDA_CALL(cudaFree(dParticles[1].velY));
  CUDA_CALL(cudaFree(dParticles[1].velZ));
  CUDA_CALL(cudaFree(dParticles[1].weight));

  CUDA_CALL(cudaFree(dCenterOfMass));
  CUDA_CALL(cudaFree(dLock));

  CUDA_CALL(cudaFreeHost(hParticles.posX));
  CUDA_CALL(cudaFreeHost(hParticles.posY));
  CUDA_CALL(cudaFreeHost(hParticles.posZ));
  CUDA_CALL(cudaFreeHost(hParticles.velX));
  CUDA_CALL(cudaFreeHost(hParticles.velY));
  CUDA_CALL(cudaFreeHost(hParticles.velZ));
  CUDA_CALL(cudaFreeHost(hParticles.weight));

  CUDA_CALL(cudaFreeHost(hCenterOfMass));

}// end of main
//----------------------------------------------------------------------------------------------------------------------
