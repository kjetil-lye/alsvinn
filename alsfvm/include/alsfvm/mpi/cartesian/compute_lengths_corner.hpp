#pragma once
#include <vector>
#include "alsfvm/types.hpp"

namespace alsfvm {
namespace mpi {
namespace cartesian {
//! computes the lengths of the segments for the mpi dispatch in a corner
std::vector<int> computeLengthsCorner(int corner, size_t dimensions,
                                      const ivec3& ghostCells)
{
    int length = 1;
    for (int i = 0; i < dimensions - 1; ++i) {
        length *= ghostCells[i];
    }
    std::vector<int> lengths(ghostCells[0], length);

    return lengths;


}
}
}
}
