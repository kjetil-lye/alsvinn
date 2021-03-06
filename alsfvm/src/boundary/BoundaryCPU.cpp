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

#include "alsfvm/boundary/BoundaryCPU.hpp"
#include "alsutils/error/Exception.hpp"
#include <cassert>
#include "alsfvm/boundary/Neumann.hpp"
#include "alsfvm/boundary/Periodic.hpp"

namespace alsfvm {
namespace boundary {
namespace {
///
/// small helper function
/// \param volume the volume to apply neumann boundary conditions to
/// \param dimensions the number of dimensions(1,2 or 3).
///
template<class BoundaryConditions>
void applyBoundary(volume::Volume& volume, const size_t dimensions,
    const size_t numberOfGhostCells,
    const grid::Grid& grid) {
    const size_t nx = volume.getTotalNumberOfXCells();
    const size_t ny = volume.getTotalNumberOfYCells();
    const size_t nz = volume.getTotalNumberOfZCells();

    if (nx < 2 * numberOfGhostCells || ((dimensions > 1)
            && ny < 2 * numberOfGhostCells) || ((dimensions > 2)
            && nz < 2 * numberOfGhostCells) ) {
        THROW("Too few cells to apply boundary condition. We got (nx, ny, nz)=("
            << nx << ", " << ny << ", " << nz << "), but numberOfGhostCells = " <<
            numberOfGhostCells
            << std::endl << std::endl <<
            "We require at least double the number of cells in each direction");
    }

    // We need these two assertions for this to work (else we need fancier indexing)
    assert(nx * sizeof(real) == volume.getScalarMemoryArea(0)->getExtentXInBytes());
    assert(ny * sizeof(real) == volume.getScalarMemoryArea(0)->getExtentYInBytes());

    // loop through variables
    for (size_t var = 0; var < volume.getNumberOfVariables(); var++) {
        auto view = volume.getScalarMemoryArea(var)->getView();

        // loop through dimensions
        for (size_t d = 0; d < dimensions; d++) {
            // i=0 represents bottom, i=1 represents top
            for (int i = 0; i < 2; i++) {
                const int side = 2 * d + i;

                if (grid.getBoundaryCondition(side) == MPI_BC) {

                    continue;
                }


                const bool zDir = (d == 2);
                const bool yDir = (d == 1);
                const bool xDir = (d == 0);
                // Either we start on the left (i == 0), or on the right(i==1)
                const size_t zStart = zDir ?
                    (i == 0 ? numberOfGhostCells : nz - numberOfGhostCells - 1) : 0;

                const size_t zEnd = zDir ?
                    (zStart + 1) : nz;

                const size_t yStart = yDir ?
                    (i == 0 ? numberOfGhostCells : ny - numberOfGhostCells - 1) : 0;

                const size_t yEnd = yDir ?
                    (yStart + 1) : ny;

                const size_t xStart = xDir ?
                    (i == 0 ? numberOfGhostCells : nx - numberOfGhostCells - 1) : 0;

                const size_t xEnd = xDir ?
                    (xStart + 1) : nx;

                for (size_t z = zStart; z < zEnd; z++) {
                    for (size_t y = yStart; y < yEnd; y++) {
                        for (size_t x = xStart; x < xEnd; x++) {

                            for (size_t ghostCell = 1; ghostCell <= numberOfGhostCells; ghostCell++) {
                                BoundaryConditions::applyBoundary(view, x, y, z, ghostCell, numberOfGhostCells,
                                    i == 1, xDir, yDir, zDir);
                            }
                        }
                    }
                }


            }
        }
    }
}
}

template<class BoundaryConditions> BoundaryCPU<BoundaryConditions>::BoundaryCPU(
    size_t numberOfGhostCells)
    : numberOfGhostCells(numberOfGhostCells) {
    // empty
}

template<class BoundaryConditions>
void BoundaryCPU<BoundaryConditions>::applyBoundaryConditions(
    volume::Volume& volume, const grid::Grid& grid) {
    applyBoundary<BoundaryConditions>(volume, grid.getActiveDimension(),
        numberOfGhostCells, grid);

}

template class BoundaryCPU < Neumann > ;
template class BoundaryCPU < Periodic >;
}
}
