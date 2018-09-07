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
#include "alsfvm/mpi/domain/DomainDecompositionParameters.hpp"
#include "alsfvm/mpi/domain/DomainDecomposition.hpp"
namespace alsfvm {
namespace mpi {
namespace domain {

//! Performs domain decomposition on a regular cartesian grid
class CartesianDecomposition : public DomainDecomposition {
public:

    //! Number of sides in 3D
    static constexpr size_t numberOfSides = 6;


    //! @todo use this enum everywhere
    //!
    //! Formula for side is \f$x + 2*y + 4*z\f$ where the coordinates are on the
    //! unit cube \f$[0,1]^3\f$.
    //!
    enum Side {
        //! \f$(-1, y, z)\f$
        X_BOTTOM = 0,

        //! \f$(1, y, z)\f$
        X_TOP = 1,

        //! \f$(x, -1, z)\f$
        Y_BOTTOM = 2,

        //! \f$(x, 1, z)\f$
        Y_TOP = 3,

        //! \f$(x, y, -1)\f$
        Z_BOTTOM = 4,

        //! \f$(x,y, 1)\f$
        Z_TOP = 5,

    };

    //! Number of corners in 3D
    static constexpr size_t numberOfCorners = 8;


    //! Also contains corner information, which is used for
    //! statistics (eg. the structure functions need the information from the corners)
    //!
    //! Corner formula is \f$x + y*2 + z*4\f$ where we consider the unit cube
    //! \f$[0,1]^3\f$.
    enum Corners {
        //! \f$(0,0,0)\f$
        LOWER_LEFT_BACK_CORNER = 0,

        //! \f$(1,0,0)\f$
        LOWER_RIGHT_BACK_CORNER = 1,

        //! \f$(0,1,0)\f$
        UPPER_LEFT_BACK_CORNER = 2,

        //! \f$(1,1,0)\f$
        UPPER_RIGHT_BACK_CORNER = 3,

        //! \f$(0, 0, 1)\f$
        LOWER_LEFT_FRONT_CORNER = 4,

        //! \f$(1, 0, 1)\f$
        LOWER_RIGHT_FRONT_CORNER = 5,

        //! \f$(0, 1, 1)\f$
        UPPER_LEFT_FRONT_CORNER = 6,

        //! \f$(1, 1, 1)\f$
        UPPER_RIGHT_FRONT_CORNER = 7,

    };

    //! Constructs a new decomposition with the parameters,
    //! uses the parameters for nx, ny, nz
    //!
    //! @param parameters used for nx, ny, nz
    CartesianDecomposition(const DomainDecompositionParameters& parameters);

    //! Constructs a new decomposition with the parameters,
    //! uses the parameters for nx, ny, nz
    //!
    //! @param nx number of cpus in x direction
    //! @param ny number of cpus in y direction
    //! @param nz number of cpus in z direction
    CartesianDecomposition(int nx, int ny, int nz);


    //! Decomposes the domain
    //!
    //! @param configuration the given mpi configuration
    //! @param grid the whole grid to decompose
    //! @return the domain information, containing the cell exchanger and
    //!         the new grid.
    virtual DomainInformationPtr decompose(ConfigurationPtr configuration,
        const grid::Grid& grid
    ) override;

    //! Decomposes the domain
    //!
    //! @param configuration the given mpi configuration
    //! @param grid the whole grid to decompose
    //! @param sides the side make the cell exchange for. The sides are ordered as
    //!     Index  |  Spatial side 1D | Spatial side 2D | Spatial side 3D
    //!     -------|------------------|-----------------|-----------------
    //!        0   |       left       |     left        |    left
    //!        1   |       right      |     right       |    right
    //!        2   |     < not used > |     bottom      |    bottom
    //!        3   |     < not used > |     top         |    top
    //!        4   |     < not used > |   < not used >  |    front
    //!        5   |     < not used > |   < not used >  |    back
    //!
    //! @param corners the corners to exchange (useful for statistics mostly,
    //!                a pure FVM simulation does not need the corner exchange,
    //!                since the default ghost cell size is large enough)
    //!
    //! @return the domain information, containing the cell exchanger and
    //!         the new grid.
    virtual DomainInformationPtr decompose(ConfigurationPtr configuration,
        const grid::Grid& grid, const std::array <bool, numberOfSides>& sides,
        const std::array<bool, numberOfCorners>& corners);
private:
    const ivec3 numberOfProcessors;
};
} // namespace domain
} // namespace mpi
} // namespace alsfvm
