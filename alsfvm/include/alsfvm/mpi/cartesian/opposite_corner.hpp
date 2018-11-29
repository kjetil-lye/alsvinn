#pragma once

namespace alsfvm {
namespace mpi {
namespace cartesian {

//! Returns the opposite corner of the given corner
int oppositeCorner(int dimensions, int corner) {
    // Recall: Corner is = 4*z + 2*y + z
    const int x = corner % 2;
    const int y = (corner / 2) % 2;
    const int z = corner / 4;

    const int newX = (x + 1) % 2;
    const int newY = dimensions > 1 ? (y + 1) % 2 : 0;
    const int newZ = dimensions > 2 ? (z + 1) % 2 : 0;

    return 4 * newZ + 2 * newY + newX;
}

}
}
}
