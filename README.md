# PCG - Project 1 - CUDA gravity particle simulation

- Author: Ondřej Vlček (xvlcek27)

## Structure
    Benchmark - reference input and output for benchmarking
    Commons   - data generation and hdf5 helper function
    Cpu       - CPU reference implementation
    Step0     - CUDA: basic implementation
    Step1     - CUDA: kernel merge
    Step2     - CUDA: shared memory
    Step3     - CUDA: GPU CoM calculation and reduction
    Step4     - CUDA: kernel paralelism
    Tests     - testing scripts and data

## Compilation on the cluster

    ml CUDA/12.2.0 GCC/11.3.0 Python/3.10.4-GCCcore-11.3.0 HDF5/1.12.2-gompi-2022a CMake/3.24.3-GCCcore-11.3.0
    mkdir build && cd build
    cmake ..
    make -j

## CLI Arguments

    1. N                       - number of particles
    2. dt                      - timestep
    3. steps                   - number of iterations
    4. threads/block           - threads per block for particle dynamics
    5. write intensity         - number of iterations between hdf5 writes
    6. reduction threads       - total threads for CoM calculation
    7. reduction threads/block - threads per block for CoM calculation
    8. input                   - input h5 file
    9. output                  - output h5 file

## Results

|   N   | CPU [s]  | Step 0 [s] | Step 1 [s] | Step 2 [s] |
|:-----:|----------|------------|------------|------------|
|  4096 | 0.492139 | 0.170714   | 0.113523   | 0.096975   |
|  8192 | 1.471328 | 0.342599   | 0.228431   | 0.190912   |
| 12288 | 2.478942 | 0.514279   | 0.343091   | 0.284863   |
| 16384 | 3.386801 | 0.685886   | 0.457310   | 0.378642   |
| 20480 | 5.059240 | 0.856444   | 0.571384   | 0.472313   |
| 24576 | 7.112179 | 1.028617   | 0.686114   | 0.565952   |
| 28672 | 9.892856 | 1.200572   | 0.800673   | 0.659676   |
| 32768 | 12.59829 | 1.371709   | 0.915028   | 0.753568   |
| 36864 | 15.54297 | 1.543931   | 1.029577   | 0.847468   |
| 40960 | 19.36099 | 1.715118   | 1.144246   | 0.941040   |
| 45056 | 23.48723 | 1.886078   | 1.258108   | 1.035073   |
| 49152 | 27.69359 | 2.056706   | 1.372358   | 1.128649   |
| 53248 | 32.63063 | 2.228323   | 1.486875   | 1.222401   |
| 57344 | 37.43660 | 3.924047   | 2.665361   | 2.196240   |
| 61440 | 42.85863 | 4.215533   | 2.863551   | 2.353482   |
| 65536 | 49.46104 | 4.506497   | 3.061437   | 2.510643   |
| 69632 | 55.14939 | 4.787021   | 3.252383   | 2.667260   |
| 73728 | 62.04446 | 5.069514   | 3.444323   | 2.822841   |
| 77824 | 69.26138 | 5.350357   | 3.635087   | 2.978833   |
| 81920 | 76.60071 | 5.632312   | 3.825689   | 3.135001   |

## Final benchmark

|    N   |  CPU [s] | GPU [s] | Speedup   | Throughput [GiB/s]  | Performance [GFLOPS] |
|:------:|:--------:|:-------:|:---------:|:-------------------:|:--------------:|
|   1024 |   1.0928 | 0.026741|    40.9   |              0.21   |         133    |
|   2048 |   0.5958 | 0.049252|    12.1   |              0.47   |         289    |
|   4096 |   0.6652 | 0.095143|     7.0   |              1.0    |         599    |
|   8192 |   1.6599 | 0.187313|     8.9   |              2.0    |         1218   |
|  16384 |   3.3655 | 0.368351|     9.1   |              4.0    |         2477   |
|  32768 |  12.7233 | 0.759010|    16.8   |              7.7    |         4809   |
|  65536 |  48.9732 | 2.509413|    19.5   |              9.3    |         5819   |
| 131072 | 195.9965 | 7.442561|    26.3   |             12.6    |         7848   |