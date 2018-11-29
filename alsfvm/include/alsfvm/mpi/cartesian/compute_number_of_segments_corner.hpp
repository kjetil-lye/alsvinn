#pragma once
#include "alsfvm/types.hpp"

namespace alsfvm {
namespace mpi {
namespace cartesian {
inline int computeNumberOfSegmentsCorner(int corner,
    size_t dimensions,
    const ivec3& numberOfCellsPerDirection,
    const ivec3& numberOfGhostCells) {
    int numberOfSegments = 1;

    for (int i = 1; i < dimensions; ++i) {
        numberOfSegments *= numberOfGhostCells[i];
    }

    return numberOfSegments;
}
}
}
}
