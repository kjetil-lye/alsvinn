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
#include "alsuq/mpi/Configuration.hpp"
#include "alsuq/types.hpp"

namespace alsuq {
namespace samples {

//! SampleInformation represents the sample being computed
//!
//!
class SampleInformation {
public:
    SampleInformation(int sampleNumber, int level, int sign);


    //! This is the global sample number (over all levels)
    int getSampleNumber() const;

    //! This is the level
    //!
    //! @note In this code, level 0 is the coarsest grid!
    int getLevel() const;

    //! Gets the sign of the sample, the sign is the sign used in the telescoping sum
    //!
    //! \f[\sum_{l}\frac{1}{M}\sum_ku^{l,+}_k-u^{l,-}_k\f]
    int getSign() const;

    alsuq::mpi::ConfigurationPtr getSpatialMpiConfiguration() const;
    void setSpatialMpiConfiguration(const alsuq::mpi::ConfigurationPtr& value);

    alsuq::mpi::ConfigurationPtr getStochasticMpiConfiguration() const;
    void setStochasticMpiConfiguration(const alsuq::mpi::ConfigurationPtr& value);

    ivec3 getMultiSpatial() const;
    void setMultiSpatial(const ivec3& value);

private:
    const int sampleNumber;
    const int level;
    const int sign;

    alsuq::mpi::ConfigurationPtr spatialMpiConfiguration = nullptr;
    alsuq::mpi::ConfigurationPtr stochasticMpiConfiguration = nullptr;

    ivec3 multiSpatial;

};
} // namespace samples
} // namespace alsuq
