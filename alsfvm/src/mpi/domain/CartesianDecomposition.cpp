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

#include "alsfvm/mpi/domain/CartesianDecomposition.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsfvm/mpi/CartesianCellExchanger.hpp"
#ifdef ALSVINN_HAVE_CUDA
    #include "alsfvm/mpi/CudaCartesianCellExchanger.hpp"
#endif
#include "alsutils/log.hpp"
#include "alsfvm/mpi/cartesian/rank_index.hpp"
#include "alsfvm/mpi/cartesian/rank_component.hpp"

namespace alsfvm {
namespace mpi {
namespace domain {

CartesianDecomposition::CartesianDecomposition(const
    DomainDecompositionParameters& parameters)
    : numberOfProcessors{parameters.getInteger("nx", 1),
          parameters.getInteger("ny", 1),
          parameters.getInteger("nz", 1)} {

}

CartesianDecomposition::CartesianDecomposition(int nx, int ny, int nz)
    : numberOfProcessors(nx, ny, nz) {

}

DomainInformationPtr CartesianDecomposition::decompose(ConfigurationPtr
    configuration,
    const grid::Grid& grid) {

    std::array<bool, CartesianDecomposition::numberOfSides> sides;
    sides.fill(true);
    std::array<bool, CartesianDecomposition::numberOfCorners> corners;
    corners.fill(false);

    return this->decompose(configuration, grid, sides, corners);

}

DomainInformationPtr CartesianDecomposition::decompose(
    alsutils::mpi::ConfigurationPtr configuration,
    const grid::Grid& grid, const std::array <bool, numberOfSides>& sides,
    const std::array<bool, numberOfCorners>& corners) {

    auto dimensions = grid.getDimensions();

    // Make sure we can evenly divide the dimensions. ie that
    // the number of processors to use in the x direction divides the number
    // of cells in the x direction, and so on.
    for (size_t i = 0; i < dimensions.size(); ++i) {
        if (dimensions[i] % numberOfProcessors[i] != 0) {
            THROW("Error in domain decompositon. In direction " << i << "\n"
                << "\tnumberOfProcessors assigned: " << numberOfProcessors[i] << "\n"
                << "\tnumberOfCells assigned      : " << dimensions[i] << "\n");
        }
    }


    int nodeNumber = configuration->getRank();
    // Find the x,y, z position of the nodeNumber.
    ivec3 nodePosition = cartesian::getCoordinates(nodeNumber, numberOfProcessors);

    ivec3 numberOfCellsPerProcessors = dimensions / numberOfProcessors;
    // startIndex for the grid
    ivec3 startIndex = numberOfCellsPerProcessors * nodePosition;


    // Geometrical position
    rvec3 startPosition = grid.getCellLengths() * startIndex + grid.getOrigin();
    rvec3 endPosition = startPosition + grid.getCellLengths() *
        numberOfCellsPerProcessors;

    // make sure it is one dimensional:
    for (size_t dim = grid.getActiveDimension(); dim < 3; ++dim) {
        endPosition[dim] = startPosition[dim];
    }

    // Find neighbours and new boundary conditions:
    ivec6 neighbours;
    std::array<boundary::Type, CartesianDecomposition::numberOfSides>
    boundaryConditions;

    for (size_t side = 0; side < 6; ++side) {
        // by default, all boundaries are now handled by MPI
        boundaryConditions[side] = boundary::Type::MPI_BC;
    }


    for (size_t side = 0; side < grid.getActiveDimension() * 2; ++side) {

        if (sides[side]) {

            if ((side % 2 == 0 && nodePosition[side / 2] == 0) || (side % 2 == 1
                    && nodePosition[side / 2] == numberOfProcessors[side / 2] - 1)) {
                // we are on the boundary

                // We should only exchange if it is periodic
                if (grid.getBoundaryCondition(int(side)) != boundary::Type::PERIODIC
                    || numberOfProcessors[side / 2] == 1 || !sides[side]) {
                    neighbours[side] = -1;
                    boundaryConditions[side] = grid.getBoundaryCondition(int(side));
                    continue;
                }
            }


            ivec3 neighbourPosition = nodePosition;
            const int i = side % 2;
            neighbourPosition[side / 2] += -(i == 0) + (i == 1);

            if (neighbourPosition[side / 2] < 0) {
                neighbourPosition[side / 2] += numberOfProcessors[side / 2];
            }

            neighbourPosition[side / 2] %= numberOfProcessors[side / 2];




            int neighbourIndex = cartesian::getRankIndex(neighbourPosition,
                    numberOfProcessors);

            if (neighbourIndex < 0) {
                THROW("NeighbourIndex got negative, this should not happen");
            }

            neighbours[side] = neighbourIndex;
        }

    }

    std::array<int, CartesianDecomposition::numberOfCorners> cornerNeighbours;
    cornerNeighbours.fill(-1);

    for (size_t corner = 0; corner < CartesianDecomposition::numberOfCorners;
        ++corner) {

        if (corners[corner]) {
            // corner position will be eg (0,1,0).
            const ivec3 cornerPosition = cartesian::getCoordinates(int(corner), {2, 2, 2});

            ivec3 neighbourPosition = nodePosition + 2 * cornerPosition - ivec3{1, 1, 1};

            // check if we are out of bounds.
            bool outOfBounds = false;

            for (size_t side = 0; side < 3; ++side) {
                if (neighbourPosition[side] < 0
                    || neighbourPosition[side] >= numberOfProcessors[side]) {

                    if (grid.getBoundaryCondition(int(side)) == boundary::PERIODIC) {
                        // we are on the boundary, but we have periodic boundary conditions, so we know this boundary
                        neighbourPosition[side] = (neighbourPosition[side] + numberOfProcessors[side]) %
                            numberOfProcessors[side];

                    } else {
                        // We are on the boundary, without periodic boundary conditions, which means,
                        // we shouldn't exchange this boundary
                        neighbourPosition[side] = -1;
                        outOfBounds = true;
                    }
                }
            }

            if (outOfBounds) {
                continue;
            }

            int neighbourIndex = cartesian::getRankIndex(neighbourPosition,
                    numberOfProcessors);

            if (neighbourIndex < 0) {
                THROW("NeighbourIndex got negative, this should not happen");
            }

            cornerNeighbours[corner] = neighbourIndex;

        }


    }



    // Create new local grid.
    auto newGrid = alsfvm::make_shared<grid::Grid>(startPosition,
            endPosition,
            numberOfCellsPerProcessors,
            boundaryConditions,
            startIndex,
            grid.getDimensions(),
            grid.getCellLengths(),
            grid.getCellMidpoints());



    alsfvm::shared_ptr<CellExchanger> cellExchanger;

    if (configuration->getPlatform() == "cpu") {
        cellExchanger.reset(new CartesianCellExchanger(configuration, neighbours,
                cornerNeighbours));
    }

#ifdef ALSVINN_HAVE_CUDA
    else {
        cellExchanger.reset(new CudaCartesianCellExchanger(configuration, neighbours,
                cornerNeighbours));
    }

#else
    else {
        THROW("CUDA not supported on this build, Tried to make a cell exchanger with CUDA");
    }

#endif

    auto information = alsfvm::make_shared<DomainInformation>(newGrid,
            cellExchanger);

    return information;
}

}
}
}
