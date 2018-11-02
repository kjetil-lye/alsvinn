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

#pragma once
#include "alsfvm/mpi/CellExchanger.hpp"
#include "alsfvm/mpi/domain/CartesianDecomposition.hpp"
#include "alsfvm/cuda/CudaMemory.hpp"
#include "alsfvm/memory/HostMemory.hpp"
#include <thrust/host_vector.h>

namespace alsfvm {
namespace mpi {

//! Does the cell exchange for cuda. This class is supposed to be a placeholder
//! for when gpu direct is not available.
class CudaCartesianCellExchanger : public CellExchanger {
public:
    constexpr static int numberOfCorners =
        domain::CartesianDecomposition::numberOfCorners;

    //! Constructs a new instance
    //!
    //! @param configuration a pointer to the current MPI configuration
    //! @param neighbours the list of processor neighbours for each side. Has
    //!                   the following format
    //!
    //! Index  |  Spatial side 1D | Spatial side 2D | Spatial side 3D
    //! -------|------------------|-----------------|-----------------
    //!    0   |       left       |     left        |    left
    //!    1   |       right      |     right       |    right
    //!    2   |     < not used > |     bottom      |    bottom
    //!    3   |     < not used > |     top         |    top
    //!    4   |     < not used > |   < not used >  |    front
    //!    5   |     < not used > |   < not used >  |    back
    CudaCartesianCellExchanger(ConfigurationPtr& configuration,
        const ivec6& neighbours,
        const std::array<int, 8>& cornerNeighbours);

    RequestContainer exchangeCells(volume::Volume& outputVolume,
        const volume::Volume& inputVolume) override;

    real max(real number) override;

    bool hasSide(int side) const;


    bool hasCorner(int corner) const;


    int getNumberOfActiveSides() const;

    ivec6 getNeighbours()  const override;
private:
    void exchangeSides(volume::Volume& outputVolume,
        const volume::Volume& inputVolume);

    void exchangeCorners(volume::Volume& outputVolume,
                        const volume::Volume& inputVolume);

    ConfigurationPtr configuration;
    ivec6 neighbours;
    const std::array<int, 8> cornerNeighbours;

    // for each variable, for each side
    std::vector<std::vector<alsfvm::shared_ptr<cuda::CudaMemory<real> > > > buffers;
    std::vector<std::vector<alsfvm::shared_ptr<cuda::CudaMemory<real> > > >
    buffersCorners;

    std::vector<std::vector<thrust::host_vector<real> > > cpuBuffersSend;
    std::vector<std::vector<thrust::host_vector<real> > > cpuBuffersReceive;

    std::vector<std::vector<thrust::host_vector<real> > > cpuBuffersSendCorners;
    std::vector<std::vector<thrust::host_vector<real> > > cpuBuffersReceiveCorners;



    void makeBuffers(const volume::Volume& inputVolume,
        const volume::Volume& outputVolume);

    void makeBuffersSides(const volume::Volume& inputVolume,
        const volume::Volume& outputVolume);

    void makeBuffersCorners(const volume::Volume& inputVolume,
        const volume::Volume& outputVolume);


    void makeStreams(const volume::Volume& inputVolume);

    void extractSides(const volume::Volume& inputVolume);
    void extractCorners(const volume::Volume& inputVolume);

    void extractMemory(const ivec3& start, const ivec3& end,
        const memory::Memory<real>& inputMemory,
        cudaStream_t stream,
        thrust::host_vector<real>& cpuBuffer,
        memory::Memory<real>& buffer);

    void extractSide(const ivec3& start, const ivec3& end,
        int side,
        const volume::Volume& inputvolume);

    void insertSides(volume::Volume& outputVolume);
    void insertMemory(const ivec3& start,
        const ivec3& end,
        memory::Memory<real>& outputMemory,
        cudaStream_t stream,
        thrust::host_vector<real>& cpuBuffer,
        memory::Memory<real>& buffer);

    void insertCorners(volume::Volume& inputVolume);

    void insertSide(const ivec3& start, const ivec3& end,
        int side,
        volume::Volume& outputVolume);


    std::vector<std::vector<cudaStream_t> > memoryStreams;
    std::vector<std::vector<cudaStream_t> > memoryStreamsCorners;

    std::vector<std::vector<RequestPtr> > receiveRequests;
    std::vector<std::vector<RequestPtr> > sendRequests;

    std::vector<std::vector<RequestPtr> > receiveRequestsCorners;
    std::vector<std::vector<RequestPtr> > sendRequestsCorners;


};


} // namespace mpi
} // namespace alsfvm
