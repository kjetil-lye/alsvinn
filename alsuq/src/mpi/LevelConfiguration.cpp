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

#include "alsuq/mpi/LevelConfiguration.hpp"
#include "alsutils/error/Exception.hpp"

namespace alsuq {
namespace mpi {

alsutils::mpi::ConfigurationPtr LevelConfiguration::getSpatialConfiguration(
    int level, int sign) const {
    auto levelMap = spatialConfigurations.find(level);

    if (levelMap != spatialConfigurations.end()) {
        auto signMap = levelMap->second.find(sign);

        if (signMap != levelMap->second.end()) {
            return signMap->second;
        } else {
            THROW("Sign not found for spatial configuration, level = "
                << level << ", sign =  " << sign);
        }

    } else {
        THROW("Level not found for spatial configuration, level = "
            << level << ", sign =  " << sign);
    }
}

alsutils::mpi::ConfigurationPtr LevelConfiguration::getStochasticConfiguration(
    int level, int sign) const {
    auto levelMap = stochasticConfiguration.find(level);

    if (levelMap != stochasticConfiguration.end()) {
        auto signMap = levelMap->second.find(sign);

        if (signMap != levelMap->second.end()) {
            return signMap->second;
        } else {
            THROW("Sign not found for stochastic configuration, level = "
                << level << ", sign =  " << sign);
        }

    } else {
        THROW("Level not found for stochastic configuration, level = "
            << level << ", sign =  " << sign);
    }
}

void LevelConfiguration::setSpatialConfiguration(int level, int sign,
    alsutils::mpi::ConfigurationPtr ptr)  {
    spatialConfigurations[level][sign] = ptr;
}

void LevelConfiguration::setStochasticConfiguration(int level, int sign,
    alsutils::mpi::ConfigurationPtr ptr) {
    stochasticConfiguration[level][sign] = ptr;
}

int LevelConfiguration::getNumberOfLevels() const {
    return stochasticConfiguration.size();
}

std::vector<int> LevelConfiguration::getSigns(int level) const {
    if (getNumberOfLevels() == 1) {
        return {{1}};
    } else {
        if (level == 0) {
            return {{1}};
        } else if (level == getNumberOfLevels() - 1) {
            return {{1}};
        } else {
            return {{-1, 1}};
        }
    }
}



}
}
