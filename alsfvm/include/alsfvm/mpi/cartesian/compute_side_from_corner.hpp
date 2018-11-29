#pragma once
#include "alsfvm/mpi/cartesian/rank_component.hpp"
namespace alsfvm {
namespace mpi {
namespace cartesian {
//! Given a corner, returns the side the corner corresponds to.
//!
//! Here the side corresponds to
//!
//!  Side  |  Spatial side 1D | Spatial side 2D | Spatial side 3D
//! -------|------------------|-----------------|-----------------
//!    0   |       left       |     left        |    left
//!    1   |       right      |     right       |    right
//!    2   |     < not used > |     bottom      |    bottom
//!    3   |     < not used > |     top         |    top
//!    4   |     < not used > |   < not used >  |    front
//!    5   |     < not used > |   < not used >  |    back
//!
//! And the corners correspond to
//!
//! Corner | Side
//! -------|------
//!    0   |
inline int computeSideFromCorner(int dimension, int corner) {
    const ivec3 coordinates = getCoordinates(corner, {2, 2, 2});


}
}
}
