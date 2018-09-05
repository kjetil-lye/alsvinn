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
#include <map>

namespace alsuq {
namespace mpi {

class LevelConfiguration {
public:

    alsuq::mpi::ConfigurationPtr getSpatialConfiguration(int level, int sign) const;

    alsuq::mpi::ConfigurationPtr getStochasticConfiguration(int level,
        int sign) const;


    void setSpatialConfiguration(int level, int sign,
        alsuq::mpi::ConfigurationPtr ptr);

    void setStochasticConfiguration(int level,
        int sign, alsuq::mpi::ConfigurationPtr ptr);

    int getNumberOfLevels() const;
    std::vector<int> getSigns(int level) const;

private:

    std::map<int, std::map<int, alsuq::mpi::ConfigurationPtr> >
    spatialConfigurations;

    std::map<int, std::map<int, alsuq::mpi::ConfigurationPtr> >
    stochasticConfiguration;


};
} // namespace mpi
} // namespace alsuq
