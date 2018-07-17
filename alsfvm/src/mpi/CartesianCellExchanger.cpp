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
    const ivec6& neighbours)
    : configuration(configuration), neighbours(neighbours) {

}

bool CartesianCellExchanger::hasSide(int side) const {

    return neighbours[side] > -1;
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

    for (int side = 0; side < 2 * dimensions; ++side) {


        for (size_t var = 0; var < inputVolume.getNumberOfVariables(); ++ var) {
            auto opposite_side = [](int s) {
                int d = s / 2;
                int i = s % 2;

                return (i + 1) % 2 + d * 2;
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


            if (hasSide(opposite_side(side))) {

                container.addRequest(Request::ireceive(*outputVolume[var],
                        1,
                        datatypesReceive[opposite_side(side)]->indexedDatatype(),
                        neighbours[opposite_side(side)],
                        side + var * 6,
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

void CartesianCellExchanger::createDataTypes(const volume::Volume&
    outputVolume,
    const volume::Volume& inputVolume) {
    const int dimensions = inputVolume.getDimensions();

    for (int side = 0; side < dimensions * 2; ++side) {
        createDataTypeSend(side, inputVolume);
        createDataTypeReceive(side, outputVolume);
    }


}

}
}
