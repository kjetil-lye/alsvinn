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

#pragma once
#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <cstdlib>
#include <cfloat>
#include <memory>
#include <complex>

#include <memory>
#include "alsutils/config.hpp"
namespace alsfvm {
using std::shared_ptr;
using std::make_shared;
using std::dynamic_pointer_cast;
}

// For CUDA we need special flags for the functions,
// for normal build, we just need to define these flags as empty.
#ifdef ALSVINN_HAVE_CUDA
    #include <cuda_runtime.h>
    #include <cuda.h>
    #include <cmath>

#else

    #define __device__
    #define __host__
#endif

#if __cplusplus <= 199711L
    #ifndef _WIN32
        #include <cassert>
        #define static_assert(x, y) assert(x)
    #endif
#endif

#include "alsutils/vec.hpp"
#include "alsutils/mat.hpp"
#ifdef ALSVINN_USE_QUADMATH
    #include <quadmath.h>
#endif


namespace alsutils {
#ifndef ALSVINN_USE_FLOAT
    typedef double real;
#else
    typedef float real;
#endif

typedef vec1<real> rvec1;
typedef vec1<int> ivec1;

typedef vec2<real> rvec2;
typedef vec2<int> ivec2;

typedef vec3<real> rvec3;
typedef vec3<int> ivec3;

typedef vec4<real> rvec4;
typedef vec4<int> ivec4;

typedef vec5<real> rvec5;
typedef vec5<int> ivec5;

typedef vec6<real> rvec6;
typedef vec6<int> ivec6;

inline __device__ __host__ ivec3 make_direction_vector(size_t direction) {
    return ivec3( direction == 0, direction == 1, direction == 2 );
}

inline __device__ __host__ ivec3 make_space_filling_vector(int nx, int ny,
    int nz) {
    return ivec3(nx > 1, ny > 1, nz > 1);
}
typedef matrix<real, 1, 1> matrix1;

typedef matrix<real, 2, 2> matrix2;
typedef matrix<real, 3, 3> matrix3;
typedef matrix<real, 4, 4> matrix4;
typedef matrix<real, 5, 5> matrix5;

template<int nsd>
struct Types {
    // empty
};

template<>
struct Types<1> {
    typedef rvec1 rvec;
    typedef ivec1 ivec;
    template<class T>
    using vec = vec1<T>;
    typedef matrix1 matrix;
};

template<>
struct Types<2> {
    typedef rvec2 rvec;
    typedef ivec2 ivec;
    template<class T>
    using vec = vec2<T>;
    typedef matrix2 matrix;
};

template<>
struct Types<3> {
    typedef rvec3 rvec;
    typedef ivec3 ivec;
    template<class T>
    using vec = vec3<T>;
    typedef matrix3 matrix;
};

template<>
struct Types<4> {
    typedef rvec4 rvec;
    typedef ivec4 ivec;
    template<class T>
    using vec = vec4<T>;
    typedef matrix4 matrix;
};

template<>
struct Types<5> {
    typedef rvec5 rvec;
    typedef ivec5 ivec;
    template<class T>
    using vec = vec5<T>;
    typedef matrix5 matrix;
};



///
/// Computes the square of x
/// \returns x*x
///
inline __host__ __device__ real square(const real& x) {
    return x * x;
}


}

#ifdef ALSVINN_USE_QUADMATH
namespace std {

inline __float128 fabs(const __float128& x) {
    return fabsq(x);
}

inline __float128 abs(const __float128& x) {
    return fabsq(x);
}

inline bool isnan(const __float128& x) {
    return isnanq(x);
}

inline bool pow(__float128 x, int b) {
    return powq(x, b);
}

inline bool pow(__float128 x, double b) {
    return powq(x, b);
}

inline bool isfinite(const __float128& x) {
    return !isinfq(x);
}

inline bool isinf(const __float128& x) {
    return isinfq(x);
}

inline std::basic_ostream<char>& operator<<(std::basic_ostream<char>& os,
    const __float128& x) {
    return (os << (double)x);
}




}
#endif
