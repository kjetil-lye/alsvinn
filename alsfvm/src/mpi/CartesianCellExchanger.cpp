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

#include "alsfvm/mpi/CartesianCellExchanger.hpp"
#include "alsutils/mpi/mpi_types.hpp"
#include "alsfvm/mpi/cartesian/number_of_segments.hpp"
#include "alsfvm/mpi/cartesian/displacements.hpp"
#include "alsfvm/mpi/cartesian/lengths.hpp"
#include "alsutils/log.hpp"


namespace alsfvm {
namespace mpi {

CartesianCellExchanger::CartesianCellExchanger(ConfigurationPtr& configuration,
    const ivec6& neighbours,
    const std::array<int, 8>& cornerNeighbours)
    : configuration(configuration), neighbours(neighbours),
      cornerNeighbours(cornerNeighbours) {

}

bool CartesianCellExchanger::hasSide(int side) const {

    return neighbours[side] > -1;
}

bool CartesianCellExchanger::hasCorner(int corner) const {
    return cornerNeighbours[corner] > -1;
}

real CartesianCellExchanger::max(real value) {
    real maximum;
    MPI_Allreduce(&value, &maximum, 1, MPI_DOUBLE,
        MPI_MAX, configuration->getCommunicator());

    return maximum;

}

ivec6 CartesianCellExchanger::getNeighbours() const {
    return neighbours;
}

RequestContainer CartesianCellExchanger::exchangeCells(volume::Volume&
    outputVolume,
    const volume::Volume& inputVolume) {
    if (datatypesReceive.size() == 0) {
        createDataTypes(outputVolume, inputVolume);
    }

    const int dimensions = outputVolume.getDimensions();

    RequestContainer container;

    const size_t numberOfSides = 2 * dimensions;
    const size_t numberOfCorners = cornerNeighbours.size();
    const size_t numberOfVariables = outputVolume.getNumberOfVariables();

    for (size_t side = 0; side < 2 * dimensions; ++side) {


        for (size_t var = 0; var < inputVolume.getNumberOfVariables(); ++ var) {
            auto oppositeSide = [](int s) {
                int d = s / 2;
                int i = s % 2;

                return size_t((i + 1) % 2 + d * 2);
            };


            if (hasSide(side)) {

                container.addRequest(Request::isend(*inputVolume[var],
                        1,
                        datatypesSend[side]->indexedDatatype(),
                        neighbours[side],
                        side + var * 6,
                        *configuration
                    ));
            }


            if (hasSide(oppositeSide(side))) {

                container.addRequest(Request::ireceive(*outputVolume[var],
                        1,
                        datatypesReceive[oppositeSide(side)]->indexedDatatype(),
                        neighbours[oppositeSide(side)],
                        side + var * 6,
                        *configuration
                    ));
            }


        }

    }


    for (int corner = 0; corner < numberOfCorners; ++corner) {


        for (size_t var = 0; var < inputVolume.getNumberOfVariables(); ++ var) {
            auto oppositeCorner = [&](int corner) {
                // Recall: Corner is = 4*z + 2*y + z
                const int x = corner % 2;
                const int y = (corner / 2) % 2;
                const int z = corner / 4;

                const int newX = (x + 1) % 2;
                const int newY = dimensions > 1 ? (y + 1) % 2 : 0;
                const int newZ = dimensions > 2 ? (z + 1) % 2 : 0;

                return 4 * newZ + 2 * newY + newX;
            };


            if (hasCorner(corner)) {

                const int tag = numberOfSides * numberOfVariables + corner + var *
                    numberOfCorners;
                container.addRequest(Request::isend(*inputVolume[var],
                        1,
                        datatypesSendCorners[corner]->indexedDatatype(),
                        cornerNeighbours[corner],
                        tag,
                        *configuration
                    ));
            }


            if (hasCorner(oppositeCorner(corner))) {

                const int tag = numberOfSides * numberOfVariables + corner + var *
                    numberOfCorners;
                container.addRequest(Request::ireceive(*outputVolume[var],
                        1,
                        datatypesReceiveCorners[oppositeCorner(corner)]->indexedDatatype(),
                        cornerNeighbours[oppositeCorner(corner)],
                        tag,
                        *configuration
                    ));
            }


        }

    }

    return container;

}

void CartesianCellExchanger::createDataTypeSend(int side,
    const volume::Volume& volume) {
    const int ghostCells = volume.getNumberOfGhostCells()[side / 2];

    const auto numberOfCellsPerDirection = volume.getSize();

    const int dimensions = volume.getDimensions();

    const int numberOfSegments = cartesian::computeNumberOfSegments(side,
            dimensions, numberOfCellsPerDirection);
    std::vector<int> displacements = cartesian::computeDisplacements(side,
            dimensions,
            numberOfCellsPerDirection,
            ghostCells,
            ghostCells);
    std::vector<int> lengths = cartesian::computeLengths(side, dimensions,
            numberOfCellsPerDirection,
            ghostCells);


    datatypesSend.push_back(MpiIndexType::makeInstance(numberOfSegments, lengths,
            displacements,
            MPI_DOUBLE));
}

void CartesianCellExchanger::createDataTypeReceive(int side,
    const volume::Volume& volume) {
    const int ghostCells = volume.getNumberOfGhostCells()[side / 2];

    const auto numberOfCellsPerDirection = volume.getSize();

    const int dimensions = volume.getDimensions();

    const int numberOfSegments = cartesian::computeNumberOfSegments(side,
            dimensions, numberOfCellsPerDirection);
    std::vector<int> displacements = cartesian::computeDisplacements(side,
            dimensions,
            numberOfCellsPerDirection,
            ghostCells,
            0);
    std::vector<int> lengths = cartesian::computeLengths(side, dimensions,
            numberOfCellsPerDirection,
            ghostCells);


    datatypesReceive.push_back(MpiIndexType::makeInstance(numberOfSegments, lengths,
            displacements,
            MPI_DOUBLE));
}


void CartesianCellExchanger::createDataTypeSendCorner(int corner,
    const volume::Volume& volume) {

    const size_t dimensions = volume.getDimensions();

    for (size_t d = 0; d < dimensions; ++d) {
        if (volume.getNumberOfGhostCells()[d] != volume.getNumberOfGhostCells()[0]) {
            THROW("In order to do the corner exchange, we require that all"
                << " dimensions use the same number of ghost cells. "
                << "\n\nGiven ghostCells = " << volume.getNumberOfGhostCells());
        }
    }

    const int ghostCells = volume.getNumberOfGhostCells()[0];

    const auto numberOfCellsPerDirection = volume.getSize();


    const int numberOfSegments = cartesian::computeNumberOfSegments(corner,
            dimensions, numberOfCellsPerDirection);
    std::vector<int> displacements = cartesian::computeDisplacements(corner,
            dimensions,
            numberOfCellsPerDirection,
            ghostCells,
            ghostCells);
    std::vector<int> lengths = cartesian::computeLengths(corner, dimensions,
            numberOfCellsPerDirection,
            ghostCells);


    datatypesSend.push_back(MpiIndexType::makeInstance(numberOfSegments, lengths,
            displacements,
            MPI_DOUBLE));
}

void CartesianCellExchanger::createDataTypeReceiveCorner(int corner,
    const volume::Volume& volume) {

    const size_t dimensions = volume.getDimensions();

    for (size_t d = 0; d < dimensions; ++d) {
        if (volume.getNumberOfGhostCells()[d] != volume.getNumberOfGhostCells()[0]) {
            THROW("In order to do the corner exchange, we require that all"
                << " dimensions use the same number of ghost cells. "
                << "\n\nGiven ghostCells = " << volume.getNumberOfGhostCells());
        }
    }

    const int ghostCells = volume.getNumberOfGhostCells().x;

    const auto numberOfCellsPerDirection = volume.getSize();


    const int numberOfSegments = cartesian::computeNumberOfSegments(corner,
            dimensions, numberOfCellsPerDirection);
    std::vector<int> displacements = cartesian::computeDisplacements(corner,
            dimensions,
            numberOfCellsPerDirection,
            ghostCells,
            0);
    std::vector<int> lengths = cartesian::computeLengths(corner, dimensions,
            numberOfCellsPerDirection,
            ghostCells);


    datatypesReceive.push_back(MpiIndexType::makeInstance(numberOfSegments, lengths,
            displacements,
            MPI_DOUBLE));
}

void CartesianCellExchanger::createDataTypes(const volume::Volume&
    outputVolume,
    const volume::Volume& inputVolume) {
    const int dimensions = inputVolume.getDimensions();

    for (int side = 0; side < dimensions * 2; ++side) {
        createDataTypeSend(side, inputVolume);
        createDataTypeReceive(side, outputVolume);
    }

    int numberOfCorners = 4 * (dimensions - 1);


    for (int corner = 0; corner < numberOfCorners; ++corner) {
        if (hasCorner(corner)) {
            createDataTypeSendCorner(corner, inputVolume);
            createDataTypeReceiveCorner(corner, outputVolume);
        } else {
            datatypesReceiveCorners.push_back(MpiIndexTypePtr());
            datatypesSendCorners.push_back(MpiIndexTypePtr());
        }
    }
}
}

}

