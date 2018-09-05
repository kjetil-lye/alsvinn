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
#include "alsuq/types.hpp"
#include <tuple>
#include "alsuq/mpi/Configuration.hpp"
#include "alsuq/samples/SampleInformation.hpp"
#include "alsuq/mpi/LevelConfiguration.hpp"

namespace alsuq {
namespace mpi {

class AbstractLoadBalancer {
public:
    virtual ~AbstractLoadBalancer() = default;

    //! @param multiSample the number of samples to run in parallel
    //!
    //! @param multiSpatial a 3 vector, for which each component is the number of processors to use in each direction.
    //!
    //! @note We require that
    //! \code{.cpp}
    //!   multiSample*multiSpatial.x*multiSpatial.y*multiSpatial.z == mpiConfigurationWorld.getNumberOfProcesses();
    //! \endcode
    //!
    //! @param mpiConfig the relevant mpiConfig
    //!
    //! \return tuple with the  mpi configurations and list of samples to compute
    //!
    virtual std::tuple
    < LevelConfiguration, std::vector<samples::SampleInformation> >
    loadBalance(
        int multiSample, ivec3 multiSpatial,
        const Configuration& mpiConfig) = 0;
};
using AbstractLoadBalancerPtr = std::shared_ptr<AbstractLoadBalancer>;
} // namespace mpi
} // namespace alsuq
