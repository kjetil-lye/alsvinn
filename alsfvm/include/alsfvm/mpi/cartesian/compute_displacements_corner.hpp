#pragma once
#include <vector>
#include "alsfvm/types.hpp"

namespace alsfvm {
namespace mpi {
namespace cartesian {
inline std::vector<int> computeDisplacementsCorner(int corner,
    size_t dimensions,
    const ivec3 ghostCells) {

    std::vector<int> displacements(size_t(ghostCells[0]), 0);

    for (int i = 0; i < ghostCells[0]; ++i) {
        displacements[i] = i * ghostCells[0];
    }

    return displacements;
}
}
}
}
