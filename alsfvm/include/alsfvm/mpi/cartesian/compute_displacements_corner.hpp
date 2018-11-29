#pragma once
#include <vector>
#include "alsfvm/types.hpp"
#include "alsfvm/mpi/cartesian/compute_number_of_segments_corner.hpp"
#include "alsfvm/mpi/cartesian/rank_component.hpp"
namespace alsfvm {
namespace mpi {
namespace cartesian {
inline std::vector<int> computeDisplacementsCorner(int corner,
    size_t dimensions,
    const ivec3& numberOfCellsPerDimension,
    const ivec3& ghostCells) {

    const int numberOfDisplacement = computeNumberOfSegmentsCorner(corner,
            dimensions,
            numberOfCellsPerDimension,
            ghostCells);

    std::vector<int> displacements(size_t(numberOfDisplacement), 0);

    const auto coordinate = getCoordinates(corner, ivec3{2, 2, 2});


    for (int segment = 0; segment < numberOfDisplacement; ++segment) {
        displacements[segment] = coordinate.x * segment
            + coordinate.y * segment * numberOfCellsPerDimension.x
            + coordinate.z * segment * numberOfCellsPerDimension.y *
            numberOfCellsPerDimension.x;
    }

    return displacements;
}
}
}
}
