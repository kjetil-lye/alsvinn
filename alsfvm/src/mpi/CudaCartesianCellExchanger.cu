/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "alsfvm/mpi/CudaCartesianCellExchanger.hpp"
#include "alsutils/mpi/safe_call.hpp"
#include "alsfvm/cuda/cuda_utils.hpp"
#include "alsutils/mpi/mpi_types.hpp"
#include "alsfvm/gpu_array.hpp"
#define L std::cout << __LINE__ << __FILE__ << std::endl;
namespace alsfvm {
namespace mpi {

namespace {

template<int numberOfSides, int numberOfVariables>
__global__ void extractSideDevice(
    gpu_array<gpu_array<memory::View<real>, numberOfSides>, numberOfVariables> output,
    gpu_array<memory::View<const real>, numberOfVariables> input,
    gpu_array<ivec3, numberOfSides> starts,
    gpu_array<ivec3, numberOfSides> ends,
    gpu_array<bool, numberOfSides> activeSides) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;

    const int side = blockIdx.y;

    if (!activeSides[side]) {
        return;
    }
    const int var = blockIdx.z;

    const int nx = ends[side].x - starts[side].x;
    const int ny = ends[side].y - starts[side].y;
    const int nz = ends[side].z - starts[side].z;

    const int x = index % nx;
    const int y = (index / nx) % ny;
    const int z = (index / nx) / ny;

    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const int inputX = x + starts[side].x;
    const int inputY = y + starts[side].y;
    const int inputZ = z + starts[side].z;

    output[var][side].at(x, y, z) = input[var].at(inputX, inputY, inputZ);

}


template<int numberOfSides, int numberOfVariables>
__global__ void insertSideDevice(gpu_array<memory::View<real>, numberOfVariables> output,
                                 gpu_array<gpu_array<memory::View<const real>, numberOfSides>, numberOfVariables> input,
                                 gpu_array<ivec3, numberOfSides> starts,
                                 gpu_array<ivec3, numberOfSides> ends,
                                 gpu_array<bool, numberOfSides> activeSides) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;


    const int side = blockIdx.y;


    if (!activeSides[side]) {
        return;
    }
    const int var = blockIdx.z;

    const int nx = ends[side].x - starts[side].x;
    const int ny = ends[side].y - starts[side].y;
    const int nz = ends[side].z - starts[side].z;

    const int x = index % nx;
    const int y = (index / nx) % ny;
    const int z = (index / nx) / ny;

    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const int outputX = x + starts[side].x;
    const int outputY = y + starts[side].y;
    const int outputZ = z + starts[side].z;

    output[var].at(outputX, outputY, outputZ) = input[var][side].at(x, y, z);
}

}
CudaCartesianCellExchanger::CudaCartesianCellExchanger(ConfigurationPtr&
    configuration, const ivec6& neighbours)
    : configuration(configuration), neighbours(neighbours) {


}

RequestContainer CudaCartesianCellExchanger::exchangeCells(
    volume::Volume& outputVolume,
    const volume::Volume& inputVolume) {

    const int dimensions = inputVolume.getDimensions();
#ifdef ALSVINN_MPI_GPU_DIRECT
    if (buffersSend.size() == 0) {
#else
    if (buffers.size() == 0) {
#endif
        makeBuffers(inputVolume);
        makeStreams(inputVolume);

        receiveRequests.resize(inputVolume.getNumberOfVariables());
        sendRequests.resize(inputVolume.getNumberOfVariables());

        for (int var = 0; var < inputVolume.getNumberOfVariables(); ++var) {
            receiveRequests[var].resize(2 * dimensions);
            sendRequests[var].resize(2 * dimensions);
        }
    }

L
    callExtractSides(inputVolume);
L
    auto oppositeSide = [&](int s) {
        int d = s / 2;
        int i = s % 2;

        return (i + 1) % 2 + d * 2;
    };
L
    RequestContainer container;

L
    for (int var = 0; var < inputVolume.getNumberOfVariables(); ++var) {
L
        for (int side = 0; side < 2 * dimensions; ++side) {
            if (hasSide(side)) {
                CUDA_SAFE_CALL(cudaStreamSynchronize(memoryStreams[var][side]));
#ifndef ALSVINN_MPI_GPU_DIRECT
                sendRequests[var][side] = (Request::isend(cpuBuffersSend[var][side],
                            cpuBuffersSend[var][side].size(),
                            alsutils::mpi::MpiTypes<real>::MPI_Real, neighbours[side],
                            var * 6 + side,
                            *configuration));
#else
                sendRequests[var][side] = (Request::isend(*buffersSend[var][side],
                            buffersSend[var][side]->getSize(),
                            alsutils::mpi::MpiTypes<real>::MPI_Real, neighbours[side],
                            var * 6 + side,
                            *configuration));
#endif
            }
L
            if (hasSide(oppositeSide(side))) {
    L
#ifndef ALSVINN_MPI_GPU_DIRECT
                receiveRequests[var][oppositeSide(side)] = Request::ireceive(
                        cpuBuffersReceive[var][oppositeSide(side)],
                        cpuBuffersReceive[var][oppositeSide(side)].size(),
                        alsutils::mpi::MpiTypes<real>::MPI_Real, neighbours[oppositeSide(side)],
                        var * 6 + side,
                        *configuration);
#else
                receiveRequests[var][oppositeSide(side)] = Request::ireceive(
                        *buffersReceive[var][oppositeSide(side)],
                        buffersReceive[var][oppositeSide(side)]->getSize(),
                        alsutils::mpi::MpiTypes<real>::MPI_Real, neighbours[oppositeSide(side)],
                        var * 6 + side,
                        *configuration);
#endif
    L
            }
        }
    }
L

    callInsertSides(outputVolume);
L
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    RequestContainer emptyContainer;
    return emptyContainer;
}

int CudaCartesianCellExchanger::getNumberOfActiveSides() const {
    int activeSides = 0;

    for (int i = 0; i < 6; ++i) {
        if (hasSide(i)) {
            activeSides++;
        }
    }

    return activeSides;
}

real CudaCartesianCellExchanger::max(real number) {
    real max;
    MPI_SAFE_CALL(MPI_Allreduce(&number, &max, 1, alsutils::mpi::MpiTypes<real>::MPI_Real, MPI_MAX,
            configuration->getCommunicator()));
    return max;
}
bool CudaCartesianCellExchanger::hasSide(int side) const {
    return neighbours[side] != -1;
}


template<int numberOfSides, int numberOfVariables>
void CudaCartesianCellExchanger::extractSide(const gpu_array<ivec3, numberOfSides>& start,
    const gpu_array<ivec3, numberOfSides>& end,
    const volume::Volume& inputVolume,
     gpu_array<bool, numberOfSides> activeSides) {



    gpu_array<gpu_array<memory::View<real>, numberOfSides>, numberOfVariables> output;
    gpu_array<memory::View<const real>, numberOfVariables> input;

L


    std::array<int, numberOfSides> sizes;
    for (int side = 0; side < numberOfSides; ++side) {

        const auto diff = end[side] - start[side];
        const int size = diff.x * diff.y * diff.z;

        sizes[side] = size;


    }
L
    // sanity check
    for(int side = 1; side < numberOfSides; ++side) {
        if (sizes[side] != sizes[side-1]) {
            THROW("We need every side to have the same number of elements for the exchange.");
        }
    }

    if (sizes[0] == 0) {
        return;
    }

L
    for (int var  = 0; var < inputVolume.getNumberOfVariables(); ++var) {
        input[var] = inputVolume.getScalarMemoryArea(var)->getView();
        for (int side = 0; side < numberOfSides; ++side) {
            if (hasSide(side)) {
#ifndef ASVINN_MPI_GPU_DIRECT
                output[var][side] = buffers[var][side]->getView();
#else
                output[var][side] = buffersSend[var][side]->getView();
#endif
            }
        }
    }


L

    const int numberOfThreads = 512;


    dim3 gridDim;
    gridDim.x = (sizes[0] + numberOfThreads - 1) / numberOfThreads;
    gridDim.y = numberOfSides;
    gridDim.z = numberOfVariables;

    extractSideDevice<numberOfSides, numberOfVariables> <<<gridDim,
                      numberOfThreads,
                      0, memoryStreams[0][0] >>> (output,
                      input,
                      start,
                      end, activeSides);
L
#ifndef ALSVINN_MPI_GPU_DIRECT
    for (int var  = 0; var < inputVolume.getNumberOfVariables(); ++var) {
        for (int side = 0; side < numberOfSides; ++side) {
            if (hasSide(side)) {
                CUDA_SAFE_CALL(cudaMemcpyAsync(cpuBuffersSend[var][side].data(),
                   buffers[var][side]->getPointer(),
                   buffers[var][side]->getSize()*sizeof(real),
                   cudaMemcpyDeviceToHost,
                   memoryStreams[0][0]));
            }
        }
    }
#endif

L


}

ivec6 CudaCartesianCellExchanger::getNeighbours() const {
    return neighbours;
}

void CudaCartesianCellExchanger::callExtractSides(const volume::Volume&
    inputVolume) {


    const int dimensions = inputVolume.getDimensions();

    switch(dimensions) {
    case 1:
        extractSides<2>(inputVolume);
        break;
    case 2:
        extractSides<4>(inputVolume);
        break;
    case 3:
        extractSides<6>(inputVolume);
        break;
    default:
        THROW("Unexpected dimension " << dimensions);
    }


}

template<int numberOfSides>
void CudaCartesianCellExchanger::callExtractSide(const gpu_array<ivec3, numberOfSides>& start,
                                                 const gpu_array<ivec3, numberOfSides>& end,
                                                 const volume::Volume& inputvolume,
                                                 gpu_array<bool, numberOfSides> activeSides) {

    const auto numberOfVariables = inputvolume.getNumberOfVariables();
    switch(numberOfVariables) {
    case 1:
        extractSide<numberOfSides, 1>(start, end, inputvolume, activeSides);
        break;
    case 2:
        extractSide<numberOfSides, 2>(start, end, inputvolume, activeSides);
        break;
    case 3:
        extractSide<numberOfSides, 3>(start, end, inputvolume, activeSides);
        break;
    case 4:
        extractSide<numberOfSides, 4>(start, end, inputvolume, activeSides);
        break;
    case 5:
        extractSide<numberOfSides, 5>(start, end, inputvolume, activeSides);
        break;
    case 6:
        extractSide<numberOfSides, 6>(start, end, inputvolume, activeSides);
        break;
    default:
        THROW("Unexpected number of variables " << numberOfVariables);
    }


}

template<int numberOfSides>
void CudaCartesianCellExchanger::extractSides(const volume::Volume&
    inputVolume) {
    const int nx = inputVolume.getTotalNumberOfXCells();
    const int ny = inputVolume.getTotalNumberOfYCells();
    const int nz = inputVolume.getTotalNumberOfZCells();

    const int ngx = inputVolume.getNumberOfXGhostCells();
    const int ngy = inputVolume.getNumberOfYGhostCells();
    const int ngz = inputVolume.getNumberOfZGhostCells();


    const int dimensions = inputVolume.getDimensions();

    gpu_array<ivec3, numberOfSides> starts;
    gpu_array<ivec3, numberOfSides> ends;
    gpu_array<bool, numberOfSides> activeSides;

    for (size_t side = 0; side < numberOfSides; ++side) {
        activeSides[side] = hasSide(side);
    }

    starts[0] = {ngx, 0, 0};
    ends[0] = {2 * ngx, ny, nz};


    starts[1] = {nx-2*ngx, 0, 0};
    ends[1] = {nx-ngx, ny, nz};


    if (dimensions > 1) {


        starts[2] = {0, ngy, 0};
        ends[2] = {nx, 2 * ngy, nz};

        starts[3] = {0, ny - 2* ngy, 0};
        ends[3] = {nx, ny-ngy, nz};


        if (dimensions > 2) {

            starts[4] = {0, 0, ngz};
            ends[4] = {nx, ny, 2 * ngz};



            starts[5] = {0, 0, nz - 2 * ngz};
            ends[5] = {nx, ny, nz - ngz};

        }

    }

    callExtractSide<numberOfSides>(starts, ends, inputVolume, activeSides);

}


void CudaCartesianCellExchanger::callInsertSides(volume::Volume&
    outputVolume) {


    const int dimensions = outputVolume.getDimensions();

    switch(dimensions) {
    case 1:
        insertSides<2>(outputVolume);
        break;
    case 2:
        insertSides<4>(outputVolume);
        break;
    case 3:
        insertSides<6>(outputVolume);
        break;
    default:
        THROW("Unexpected dimension " << dimensions);
    }


}

template<int numberOfSides>
void CudaCartesianCellExchanger::callInsertSide(const gpu_array<ivec3, numberOfSides>& start,
                                                 const gpu_array<ivec3, numberOfSides>& end,
                                                 volume::Volume& outputVolume,
                                                 gpu_array<bool, numberOfSides> activeSides) {

    const auto numberOfVariables = outputVolume.getNumberOfVariables();
    switch(numberOfVariables) {
    case 1:
        insertSide<numberOfSides, 1>(start, end, outputVolume, activeSides);
        break;
    case 2:
        insertSide<numberOfSides, 2>(start, end, outputVolume, activeSides);
        break;
    case 3:
        insertSide<numberOfSides, 3>(start, end, outputVolume, activeSides);
        break;
    case 4:
        insertSide<numberOfSides, 4>(start, end, outputVolume, activeSides);
        break;
    case 5:
        insertSide<numberOfSides, 5>(start, end, outputVolume, activeSides);
        break;
    case 6:
        insertSide<numberOfSides, 6>(start, end, outputVolume, activeSides);
        break;
    default:
        THROW("Unexpected number of variables " << numberOfVariables);
    }


}


template<int numberOfSides, int numberOfVariables>
void CudaCartesianCellExchanger::insertSide(const gpu_array<ivec3, numberOfSides>& start,
                                            const gpu_array<ivec3, numberOfSides>& end,
                                            volume::Volume& outputVolume,
                                            gpu_array<bool, numberOfSides> activeSides) {


    gpu_array<gpu_array<memory::View<const real>, numberOfSides>, numberOfVariables> input;
    gpu_array<memory::View<real>, numberOfVariables> output;

L


    std::array<int, numberOfSides> sizes;
    for (int side = 0; side < numberOfSides; ++side) {

        const auto diff = end[side] - start[side];
        const int size = diff.x * diff.y * diff.z;

        sizes[side] = size;


    }
L
    // sanity check
    for(int side = 1; side < numberOfSides; ++side) {
        if (sizes[side] != sizes[side-1]) {
            THROW("We need every side to have the same number of elements for the exchange.");
        }
    }

    if (sizes[0] == 0) {
        return;
    }
L

    for (int var  = 0; var < outputVolume.getNumberOfVariables(); ++var) {
        output[var] = outputVolume.getScalarMemoryArea(var)->getView();
        for (int side = 0; side < numberOfSides; ++side) {

            if (hasSide(side)) {
#ifndef ALSVINN_MPI_GPU_DIRECT
                input[var][side] = buffers[var][side]->getConstView();
#else
                input[var][side] = buffersReceive[var][side]->getConstView();
#endif
            }

        }
    }
L

#ifndef ALSVINN_MPI_GPU_DIRECT

    for (int var  = 0; var < outputVolume.getNumberOfVariables(); ++var) {
        for (int side = 0; side < numberOfSides; ++side) {
            if (hasSide(side)) {
                sendRequests[var][side]->wait();
                receiveRequests[var][side]->wait();

                CUDA_SAFE_CALL(cudaMemcpyAsync(buffers[var][side]->getPointer(),
                    cpuBuffersReceive[var][side].data(),
                    buffers[var][side]->getSize()*sizeof(real),
                    cudaMemcpyHostToDevice,
                    memoryStreams[0][0]));
            }
        }
    }
#else
    for (int var  = 0; var < outputVolume.getNumberOfVariables(); ++var) {
        for (int side = 0; side < numberOfSides; ++side) {
            if (hasSide(side)) {
                sendRequests[var][side]->wait();
                receiveRequests[var][side]->wait();
            }
        }
    }
#endif
L


    const int numberOfThreads = 512;

L
    dim3 gridDim;
    gridDim.x = (sizes[0] + numberOfThreads - 1) / numberOfThreads;
    gridDim.y = numberOfSides;
    gridDim.z = numberOfVariables;

    insertSideDevice<numberOfSides, numberOfVariables> <<< gridDim,
                         numberOfThreads,
                         0, memoryStreams[0][0] >>> (
                             output,
                             input,
                             start,
                             end, activeSides);

L
}

template<int numberOfSides>
void CudaCartesianCellExchanger::insertSides( volume::Volume& outputVolume) {
    const int nx = outputVolume.getTotalNumberOfXCells();
    const int ny = outputVolume.getTotalNumberOfYCells();
    const int nz = outputVolume.getTotalNumberOfZCells();

    const int ngx = outputVolume.getNumberOfXGhostCells();
    const int ngy = outputVolume.getNumberOfYGhostCells();
    const int ngz = outputVolume.getNumberOfZGhostCells();


    const int dimensions = outputVolume.getDimensions();

    gpu_array<ivec3, numberOfSides> starts;
    gpu_array<ivec3, numberOfSides> ends;
    gpu_array<bool, numberOfSides> activeSides;

    for (size_t side = 0; side < numberOfSides; ++side) {
        activeSides[side] = hasSide(side);
    }


    starts[0] = {0, 0, 0};
    ends[0] =  {ngx, ny, nz};

    starts[1] = {nx - ngx, 0, 0};
    ends[1] = {nx, ny, nz};

    if (dimensions > 1) {

       starts[2] = {0, 0, 0};
       ends[2] = {nx, ngy, nz};

       starts[3] = {0, ny - ngy, 0};
       ends[3] = {nx, ny, nz};
       if (dimensions > 2 ) {
            starts[4] = {0, 0, 0};
            ends[4] = {nx, ny, ngz};

            starts[5] = {0, 0, nz - ngz};
            ends[5] = {nx, ny, nz};
        }
    }

    callInsertSide<numberOfSides>(starts, ends, outputVolume, activeSides);

}


void CudaCartesianCellExchanger::makeStreams(const volume::Volume&
    inputVolume) {
    memoryStreams.resize(inputVolume.getNumberOfVariables());

    for (int var = 0; var < inputVolume.getNumberOfVariables(); ++var) {
        memoryStreams[var].resize(6);
        int dimensions = inputVolume.getDimensions();

        for (int side = 0; side < 2 * dimensions; ++side) {
            CUDA_SAFE_CALL(cudaStreamCreate(&memoryStreams[var][side]));
            //memoryStreams[var][side]=0;
        }
    }
}

void CudaCartesianCellExchanger::makeBuffers(const volume::Volume&
    inputVolume) {
#ifndef ALSVINN_MPI_GPU_DIRECT
    buffers.resize(inputVolume.getNumberOfVariables());
    cpuBuffersSend.resize(buffers.size());
    cpuBuffersReceive.resize(buffers.size());
#else
    buffersSend.resize(inputVolume.getNumberOfVariables());
    buffersReceive.resize(buffersSend.size());
#endif
    for (int var = 0; var < inputVolume.getNumberOfVariables(); ++var) {
#ifndef ALSVINN_MPI_GPU_DIRECT
        buffers[var].resize(6);
        cpuBuffersSend[var].resize(6);
        cpuBuffersReceive[var].resize(6);
#else
        buffersSend[var].resize(6);
        buffersReceive[var].resize(6);
#endif

        for (int side = 0; side < 6; ++side) {
            if (hasSide(side)) {
                const int nx = (side > 1) * inputVolume.getTotalNumberOfXCells() +
                    (side < 2) * inputVolume.getNumberOfXGhostCells();

                const int ny = (side != 2) * (side != 3) * inputVolume.getTotalNumberOfYCells()
                    +
                    ((side == 2) + (side == 3)) * inputVolume.getNumberOfYGhostCells();


                const int nz = (side != 4) * (side != 5) * inputVolume.getTotalNumberOfZCells()
                    +
                    ((side == 4) + (side == 5)) * inputVolume.getNumberOfZGhostCells();
#ifndef ALSVINN_MPI_GPU_DIRECT
                buffers[var][side] = alsfvm::make_shared<alsfvm::cuda::CudaMemory<real>>(nx, ny,
                        nz);

                cpuBuffersSend[var][side].resize(nx * ny * nz, 0);
                cpuBuffersReceive[var][side].resize(nx * ny * nz, 0);
#else
                buffersSend[var][side] = alsfvm::make_shared<alsfvm::cuda::CudaMemory<real>>(nx, ny,
                        nz);

                buffersReceive[var][side] = alsfvm::make_shared<alsfvm::cuda::CudaMemory<real>>(nx, ny,
                        nz);
#endif
                //alsfvm::make_shared<alsfvm::memory::HostMemory<real>>(nx, ny, nz);
            }
        }
    }
}
}
}
