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
#include <cassert>

namespace  alsfvm {
namespace memory {

///
/// View is raw view to the memory area. It will only contain the most
/// basic functionality. This is ideal for use in an inner loop.
///
/// The View class is not meant for memory management, it will *not* delete
/// its pointer on scope exit.
///
template<class T>
class View {
public:




    ///
    /// \brief View constructs the View
    /// \param pointer raw pointer to the data.
    /// \param nx the number of elements in x direction (in number of T's)
    ///        (used for bounds checking)
    /// \param ny the number of elements in y direction (in number of T's)
    ///        (used for bounds checking)
    /// \param nz the number of elements in z direction (in number of T's)
    ///        (used for bounds checking)
    /// \param extentXInBytes the extent in X direction (used for indexing)
    /// \param extentYInBytes the extent in Y direction (used for indexing)
    ///
    __device__ __host__ View(T* pointer,
        size_t nx,
        size_t ny,
        size_t nz,
        size_t extentXInBytes,
        size_t extentYInBytes)
        : nx(nx), ny(ny), nz(nz), pointer(pointer), extentXInBytes(extentXInBytes),
          extentYInBytes(extentYInBytes) {
        // empty
    }

    __device__ __host__ View(const View<T>& other)
        : View(other.pointer, other.nx, other.ny, other.nz, other.extentXInBytes,
              other.extentYInBytes) {
    }


    __device__ __host__ View<T>& operator=(const View<T>& other) {
        this->pointer = other.pointer;
        this->nx = other.nx;
        this->ny = other.ny;
        this->nz = other.nz;

        this->extentXInBytes = other.extentXInBytes;
        this->extentYInBytes = other.extentYInBytes;

        return *this;

    }


    __device__ __host__  View() : View(nullptr, 0, 0, 0, 0, 0) {

    }

    ///
    /// \brief at returns a reference to the element at the given location
    /// \param x the x coordinate
    /// \param y the y coordinate
    /// \param z the z coordinate
    /// \return the reference to the element at the given location.
    ///
    __device__ __host__  T& at(size_t x, size_t y, size_t z) {
        assert(x < nx);
        assert(y < ny);
        assert(z < nz);
        return pointer[index(x, y, z)];
    }

    ///
    /// \brief at returns a reference to the element at the given location (const version)
    /// \param x the x coordinate
    /// \param y the y coordinate
    /// \param z the z coordinate
    /// \return the reference to the element at the given location.
    ///
    __device__ __host__  const T& at(size_t x, size_t y, size_t z) const {
        assert(x < nx);
        assert(y < ny);
        assert(z < nz);
        return pointer[index(x, y, z)];
    }

    ///
    /// \brief at returns the reference to the element index by the single value index
    /// \param index
    /// \return
    ///
    __device__ __host__  T& at(size_t index) {
        return pointer[index];
    }

    ///
    /// \brief at returns the reference to the element index by the single value index
    /// \param index
    /// \return
    ///
    __device__ __host__  const T& at(size_t index) const {
        return pointer[index];
    }

    __device__ __host__ size_t size() const {
        return nx * ny * nz;
    }

    ///
    /// \brief index computes the linear index of the given cell
    /// \param x the x coordinate
    /// \param y the y coordinate
    /// \param z the z coordinate
    /// \return the linear index
    ///
    __device__ __host__ size_t index(size_t x, size_t y, size_t z) const {
        return z * nx * ny + y * nx + x ;
    }

    __device__ __host__ size_t getNumberOfXCells() const {
        return nx;
    }

    __device__ __host__ size_t getNumberOfYCells() const {
        return ny;
    }

    __device__ __host__ size_t getNumberOfZCells() const {
        return nz;
    }

    size_t nx;
    size_t ny;
    size_t nz;
private:
    T* pointer;

    size_t extentXInBytes;
    size_t extentYInBytes;

};
}

}
