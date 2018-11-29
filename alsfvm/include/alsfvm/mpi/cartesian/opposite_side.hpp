#pragma once
#include "alsfvm/types.hpp"
namespace alsfvm {
namespace mpi {
namespace cartesian {

//! Finds the opposite side of the given side.
//!
inline size_t oppositeSide(size_t s) {
    size_t d = s / 2;
    size_t i = s % 2;

    return size_t((i + 1) % 2 + d * 2);

}
}
}
}
