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

#include "alsfvm/memory/HostMemory.hpp"
#include "alsfvm/memory/memory_utils.hpp"
#include <cassert>
#include <algorithm>
#include "alsutils/error/Exception.hpp"
#include "alsutils/log.hpp"
#include "alsutils/debug/stacktrace.hpp"
#include "alsutils/config.hpp"

#define CHECK_SIZE_AND_HOST(x) { \
    if (!x.isOnHost()) {\
        THROW(#x << " is not on host."); \
    } \
    if (this->getSize() != x.getSize()) { \
        THROW("Size mismatch: \n\tthis->getSize() = " << this->getSize() <<"\n\t"<<#x<<".getSize() = " << x.getSize()); \
    } \
}
namespace alsfvm {
namespace memory {

template<class T> HostMemory<T>::HostMemory(size_t nx, size_t ny, size_t nz)
    : Memory<T>(nx, ny, nz), data(nx * ny * nz, 42) {
#ifdef ALSVINN_PRINT_MEMORY_ALLOCATIONS
    const  size_t size = nx * ny * nz * sizeof(T);
    const double sizeGb = size / 1000000000.;
    ALSVINN_LOG(INFO, "CPU allocating " << size << " bytes (" << sizeGb <<
        " GB), in " << alsutils::debug::getShortStacktrace());
#endif

}

template<class T>
std::shared_ptr<Memory<T> > HostMemory<T>::makeInstance() const {
    std::shared_ptr<Memory<T>> memoryArea;

    memoryArea.reset(new HostMemory(this->nx, this->ny, this->nz));

    return memoryArea;
}

template<class T>
bool HostMemory<T>::isOnHost() const {
    return true;
}

template<class T>
void HostMemory<T>::copyFrom(const Memory<T>& other) {
    CHECK_SIZE_AND_HOST(other);

    std::copy(other.data(), other.data() + this->getSize(), data.begin());
}

template<class T>
T* HostMemory<T>::getPointer() {
    return data.data();
}

template<class T>
const T* HostMemory<T>::getPointer() const {
    return data.data();
}

template<class T>
void HostMemory<T>::copyToHost(T* bufferPointer, size_t bufferLength) const {
    assert(bufferLength >= Memory<T>::getSize());
    std::copy(data.begin(), data.end(), bufferPointer);
}

template<class T>
void HostMemory<T>::copyFromHost(const T* bufferPointer, size_t bufferLength) {
    const size_t sizeToCopy = std::min(bufferLength, Memory<T>::getSize());
    std::copy(bufferPointer, bufferPointer + sizeToCopy, data.begin());
}



///
/// Adds the other memory area to this one
/// \param other the memory area to add from
///
template <class T>
void HostMemory<T>::operator+=(const Memory<T>& other) {
    if (!other.isOnHost()) {
        THROW("Memory not on host");
    }

    auto pointer = other.getPointer();
    #pragma omp parallel for simd

    for (int i = 0; i < int(data.size()); ++i) {
        data[i] += pointer[i];
    }
}

///
/// Mutliplies the other memory area to this one
/// \param other the memory area to multiply from
///
template <class T>
void HostMemory<T>::operator*=(const Memory<T>& other) {
    if (!other.isOnHost()) {
        THROW("Memory not on host");
    }

    if (other.getSize() != this->getSize()) {
        THROW("Memory size not the same");
    }

    auto pointer = other.getPointer();
    #pragma omp parallel for simd

    for (int i = 0; i < int(data.size()); ++i) {
        data[i] *= pointer[i];
    }
}

///
/// Subtracts the other memory area to this one
/// \param other the memory area to subtract from
///
template <class T>
void HostMemory<T>::operator-=(const Memory<T>& other) {
    if (!other.isOnHost()) {
        THROW("Memory not on host");
    }

    if (other.getSize() != this->getSize()) {
        THROW("Memory size not the same");
    }

    auto pointer = other.getPointer();
    #pragma omp parallel for simd

    for (int i = 0; i < int(data.size()); ++i) {
        data[i] -= pointer[i];
    }
}

///
/// Divides the other memory area to this one
/// \param other the memory area to divide from
///
template <class T>
void HostMemory<T>::operator/=(const Memory<T>& other) {
    if (!other.isOnHost()) {
        THROW("Memory not on host");
    }

    if (other.getSize() != this->getSize()) {
        THROW("Memory size not the same");
    }

    auto pointer = other.getPointer();
    #pragma omp parallel for simd

    for (int i = 0; i < int(data.size()); ++i) {
        data[i] /= pointer[i];
    }
}

///
/// Adds the scalar to each component
/// \param scalar the scalar to add
///
template <class T>
void HostMemory<T>::operator+=(real scalar) {

    #pragma omp parallel for simd

    for (int i = 0; i < int(data.size()); ++i) {
        data[i] += scalar;
    }
}

///
/// Multiplies the scalar to each component
/// \param scalar the scalar to multiply
///
template <class T>
void HostMemory<T>::operator*=(real scalar) {
    #pragma omp parallel for simd

    for (int i = 0; i < int(data.size()); ++i) {
        data[i] *= scalar;
    }
}

///
/// Subtracts the scalar from each component
/// \param scalar the scalar to subtract
///
template <class T>
void HostMemory<T>::operator-=(real scalar) {
    #pragma omp parallel for simd

    for (int i = 0; i < int(data.size()); ++i) {
        data[i] -= scalar;
    }
}

///
/// Divides the each component by the scalar
/// \param scalar the scalar to divide
///
template <class T>
void HostMemory<T>::operator/=(real scalar) {
    #pragma omp parallel for simd

    for (int i = 0; i < int(data.size()); ++i) {
        data[i] /= scalar;
    }
}

template <class T>
void HostMemory<T>::makeZero() {
    #pragma omp parallel for simd

    for (int i = 0; i < int(data.size()); ++i) {
        data[i] = 0;
    }
}

template <class T>
void HostMemory<T>::copyInternalCells(size_t startX, size_t endX, size_t startY,
    size_t endY, size_t startZ, size_t endZ, T* output, size_t outputSize) {


    const size_t nx = this->nx;
    const size_t ny = this->ny;
    const size_t numberOfY = endY - startY;
    const size_t numberOfX = endX - startX;

    for (size_t z = startZ; z < endZ; z++) {
        for (size_t y = startY; y < endY; y++) {
            for (size_t x = startX; x < endX; x++) {
                size_t indexIn = z * nx * ny + y * nx + x;
                size_t indexOut = (z - startZ) * numberOfX * numberOfY
                    + (y - startY) * numberOfX + (x - startX);
                output[indexOut] = data[indexIn];
            }
        }
    }
}

template<class T>
void HostMemory<T>::addLinearCombination(T a1,
    T a2, const Memory<T>& v2,
    T a3, const Memory<T>& v3,
    T a4, const Memory<T>& v4,
    T a5, const Memory<T>& v5) {
    CHECK_SIZE_AND_HOST(v2);
    CHECK_SIZE_AND_HOST(v3);
    CHECK_SIZE_AND_HOST(v4);
    CHECK_SIZE_AND_HOST(v5);
    const auto& d1 = data;
    auto d2 = v2.getPointer();
    auto d3 = v3.getPointer();
    auto d4 = v4.getPointer();
    auto d5 = v5.getPointer();
    #pragma omp parallel for

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = a1 * d1[i] + a2 * d2[i] + a3 * d3[i] + a4 * d4[i] + a5 * d5[i];
    }
}

template<class T>
void HostMemory<T>::addPower(const Memory<T>& other, double power) {
    CHECK_SIZE_AND_HOST(other);
    #pragma omp parallel for

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += std::pow(other[i], power);
    }
}

template<class T>
void HostMemory<T>::addPower(const Memory<T>& other, double power,
    double factor) {
    CHECK_SIZE_AND_HOST(other);
    #pragma omp parallel for

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += factor * std::pow(other[i], power);
    }
}

template<class T>
void HostMemory<T>::subtractPower(const Memory<T>& other, double power) {
    CHECK_SIZE_AND_HOST(other);
    #pragma omp parallel for

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= std::pow(other[i], power);
    }
}

template<class T>
std::shared_ptr<Memory<T> > HostMemory<T>::getHostMemory() {
    return this->shared_from_this();
}

template<class T>
const std::shared_ptr<const Memory<T> > HostMemory<T>::getHostMemory() const {
    return this->shared_from_this();

}

template<class T>
real HostMemory<T>::getTotalVariation(int p, const ivec3& start,
    const ivec3& end) const {
    // See http://www.ams.org/journals/tran/1933-035-04/S0002-9947-1933-1501718-2/S0002-9947-1933-1501718-2.pdf
    //
    const size_t nx = this->nx;
    const size_t ny = this->ny;
    const size_t nz = this->nz;

    if (nz > 1 ) {
        THROW("Not supported for 3d yet");
    }

    const size_t startX = start.x + 1;
    const size_t startY = start.y + (ny > 1 ? 1 : 0);
    T bv = 0;

    for (int z = 0; z < end.z; z++) {
        for (int y = startY; y < end.y; y++) {
            for (int x = startX; x < end.x; x++) {
                int index = z * nx * ny + y * nx + x;
                int indexXLeft = z * nx * ny + y * nx + (x - 1);

                int yBottom = ny > 1 ? y - 1 : 0;


                int indexYLeft = z * nx * ny + yBottom * nx + x;


                bv += std::pow(std::sqrt(std::pow(data[index]
                                - data[indexYLeft], 2) + std::pow(data[index]
                                - data[indexXLeft], 2)), p);
            }
        }
    }

    return bv;

}

template<class T>
real HostMemory<T>::getTotalVariation(int direction, int p, const ivec3& start,
    const ivec3& end) const {
    // See http://www.ams.org/journals/tran/1933-035-04/S0002-9947-1933-1501718-2/S0002-9947-1933-1501718-2.pdf
    //

    auto directionVector = make_direction_vector(direction);

    const size_t ny = this->ny;
    const size_t nz = this->nz;

    if (direction > (1 + (ny > 1) + (nz > 1))) {
        THROW("direction = " << direction << " is bigger than current dimension");
    }

    const int startX = start.x + directionVector.x;
    const int startY = start.y + directionVector.y;
    const int startZ = start.z + directionVector.z;
    T bv = 0;

    const auto view = this->getView();


    for (int z = startZ; z < end.z; z++) {
        for (int y = startY; y < end.y; y++) {
            for (int x = startX; x < end.x; x++) {

                auto positionLeft = ivec3(x, y, z) - directionVector;

                bv += std::pow(std::abs(view.at(x, y, z) - view.at(positionLeft.x,
                                positionLeft.y, positionLeft.z)), p);
            }
        }
    }

    return bv;

}


INSTANTIATE_MEMORY(HostMemory)
} // namespace memory
} // namespace alsfvm

