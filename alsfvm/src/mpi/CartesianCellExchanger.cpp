#include "alsfvm/mpi/CartesianCellExchanger.hpp"
#include "alsutils/mpi/mpi_types.hpp"
#include "alsfvm/mpi/cartesian/number_of_segments.hpp"
#include "alsfvm/mpi/cartesian/displacements.hpp"
#include "alsfvm/mpi/cartesian/lengths.hpp"


namespace alsfvm { namespace mpi {

CartesianCellExchanger::CartesianCellExchanger(ConfigurationPtr &configuration,
                                               const ivec6 &neighbours)
    : configuration(configuration), neighbours(neighbours)
{

}

bool CartesianCellExchanger::hasSide(int side) const
{
    return neighbours[side] > -1;
}

RequestContainer CartesianCellExchanger::exchangeCells(volume::Volume &outputVolume,
                                           const volume::Volume &inputVolume)
{
    if (datatypesReceive.size() == 0) {
        createDataTypes(outputVolume);
    }

    const int dimensions = outputVolume.getDimensions();

    RequestContainer container;
    for (int side = 0; side < 2 * dimensions; ++side) {
        if (hasSide(side)) {
            for (int var = 0; var < inputVolume.getNumberOfVariables(); ++ var) {
                container.addRequest(Request::isend(*inputVolume[var],
                                                    1,
                                                    datatypesSend[side].indexedDatatype(),
                                                    neighbours[side],
                                                    0,
                                                    *configuration
                                                    ));

                container.addRequest(Request::ireceive(*outputVolume[var],
                                                       1,
                                                       datatypesReceive[side].indexedDatatype(),
                                                       neighbours[side],
                                                       0,
                                                       *configuration
                                                       ));

            }
        }
    }

    return container;

}

void CartesianCellExchanger::createDataTypeSend(int side, const volume::Volume &volume)
{
    const int ghostCells = volume.getNumberOfGhostCells()[side / 2];

    const auto numberOfCellsPerDirection = volume.getSize();

    const int dimensions = volume.getDimensions();

    const int numberOfSegments = cartesian::computeNumberOfSegments(side, dimensions, numberOfCellsPerDirection);
    std::vector<int> displacements = cartesian::computeDisplacements(side, dimensions,
                                                                     numberOfCellsPerDirection,
                                                                     ghostCells,
                                                                     ghostCells);
    std::vector<int> lengths = cartesian::computeLengths(side, dimensions,
                                                         numberOfCellsPerDirection,
                                                         ghostCells);



    datatypesSend.push_back(MpiIndexType(numberOfSegments, lengths, displacements,
                                  alsutils::mpi::MpiTypes<real>::MPI_Real));
}

void CartesianCellExchanger::createDataTypeReceive(int side, const volume::Volume &volume)
{
    const int ghostCells = volume.getNumberOfGhostCells()[side / 2];

    const auto numberOfCellsPerDirection = volume.getSize();

    const int dimensions = volume.getDimensions();

    const int numberOfSegments = cartesian::computeNumberOfSegments(side, dimensions, numberOfCellsPerDirection);
    std::vector<int> displacements = cartesian::computeDisplacements(side, dimensions,
                                                                     numberOfCellsPerDirection,
                                                                     ghostCells,
                                                                     0);
    std::vector<int> lengths = cartesian::computeLengths(side, dimensions,
                                                         numberOfCellsPerDirection,
                                                         ghostCells);


    datatypesReceive.push_back(MpiIndexType(numberOfSegments, lengths, displacements,
                                  alsutils::mpi::MpiTypes<real>::MPI_Real));
}

void CartesianCellExchanger::createDataTypes(const volume::Volume &volume)
{
    const int dimensions = volume.getDimensions();

    for (int side = 0; side < dimensions * 2; ++side) {
        createDataTypeSend(side, volume);
        createDataTypeReceive(side, volume);
    }


}

}
}