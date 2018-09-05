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

#include "alsuq/stats/Statistics.hpp"
#include "alsuq/mpi/utils.hpp"
namespace alsuq {
namespace stats {

void Statistics::write(const alsfvm::volume::Volume& conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    computeStatistics(conservedVariables, extraVariables, grid,
        timestepInformation);
}

void Statistics::setMpiSpatialConfiguration(const mpi::ConfigurationPtr&
    value) {
    mpiSpatialConfiguration = value;
}

void Statistics::setMpiStochasticConfiguration(const mpi::ConfigurationPtr&
    value) {
    mpiStochasticConfiguration = value;
}

mpi::ConfigurationPtr Statistics::getMpiSpatialConfiguration() const {
    if (!mpiSpatialConfiguration) {
        THROW("Trying to get mpiSpatialConfiguration, but it has not been set");
    }

    return mpiSpatialConfiguration;
}

mpi::ConfigurationPtr Statistics::getMpiStochasticConfiguration() const {
    if (!mpiStochasticConfiguration) {
        THROW("Trying to get mpiStochasticConfiguration, but it has not been set");
    }

    return mpiStochasticConfiguration;
}



}
}
