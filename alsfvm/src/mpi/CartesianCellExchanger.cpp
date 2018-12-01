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
#include "alsfvm/mpi/cartesian/opposite_side.hpp"
#include "alsfvm/mpi/cartesian/opposite_corner.hpp"
#include "alsfvm/mpi/cartesian/compute_displacements_corner.hpp"
#include "alsfvm/mpi/cartesian/compute_lengths_corner.hpp"
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

    return neighbours[size_t(side)] > -1;
}

bool CartesianCellExchanger::hasCorner(int corner) const {
    return cornerNeighbours[size_t(corner)] > -1;
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



            if (hasSide(side)) {

                container.addRequest(Request::isend(*inputVolume[var],
                        1,
                        datatypesSend[side]->indexedDatatype(),
                        neighbours[side],
                        side + var * 6,
                        *configuration
                    ));
            }


            if (hasSide(cartesian::oppositeSide(side))) {

                container.addRequest(Request::ireceive(*outputVolume[var],
                        1,
                        datatypesReceive[cartesian::oppositeSide(side)]->indexedDatatype(),
                        neighbours[cartesian::oppositeSide(side)],
                        side + var * 6,
                        *configuration
                    ));
            }


        }

    }


    for (size_t corner = 0; corner < numberOfCorners; ++corner) {


        for (size_t var = 0; var < inputVolume.getNumberOfVariables(); ++ var) {


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


            if (hasCorner(cartesian::oppositeCorner(dimensions, corner))) {

                const int tag = numberOfSides * numberOfVariables + corner + var *
                    numberOfCorners;

                container.addRequest(Request::ireceive(*outputVolume[var],
                        1,
                        datatypesReceiveCorners[cartesian::oppositeCorner(dimensions,
                                corner)]->indexedDatatype(),
                        cornerNeighbours[cartesian::oppositeCorner(dimensions, corner)],
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

    const int numberOfSegments = cartesian::computeNumberOfSegmentsCorner(corner,
            dimensions,
            numberOfCellsPerDirection,
            volume.getNumberOfGhostCells());




    std::vector<int> displacements = cartesian::computeDisplacementsCorner(corner,
            dimensions,
            numberOfCellsPerDirection,
            ghostCells);

    std::vector<int> lengths = cartesian::computeLengthsCorner(corner, dimensions,
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


    const int numberOfSegments = cartesian::computeNumberOfSegmentsCorner(corner,
            dimensions,
            numberOfCellsPerDirection,
            volume.getNumberOfGhostCells());

    std::vector<int> displacements = cartesian::computeDisplacementsCorner(corner,
            dimensions,
            numberOfCellsPerDirection,
            ghostCells);

    std::vector<int> lengths = cartesian::computeLengthsCorner(corner, dimensions,
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

