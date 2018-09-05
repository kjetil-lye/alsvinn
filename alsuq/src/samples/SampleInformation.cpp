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

#include "alsuq/samples/SampleInformation.hpp"

namespace alsuq {
namespace samples {

SampleInformation::SampleInformation(int sampleNumber, int level, int sign)
    : sampleNumber(sampleNumber), level(level), sign(sign) {

}

int SampleInformation::getSampleNumber() const {
    return sampleNumber;
}

int SampleInformation::getLevel() const {
    return level;
}

int SampleInformation::getSign() const {
    return sign;
}

alsuq::mpi::ConfigurationPtr SampleInformation::getSpatialMpiConfiguration() const
{
    return spatialMpiConfiguration;
}

void SampleInformation::setSpatialMpiConfiguration(const alsuq::mpi::ConfigurationPtr &value)
{
    spatialMpiConfiguration = value;
}

alsuq::mpi::ConfigurationPtr SampleInformation::getStochasticMpiConfiguration() const
{
    return stochasticMpiConfiguration;
}

void SampleInformation::setStochasticMpiConfiguration(const alsuq::mpi::ConfigurationPtr &value)
{
    stochasticMpiConfiguration = value;
}

ivec3 SampleInformation::getMultiSpatial() const
{
    return multiSpatial;
}

void SampleInformation::setMultiSpatial(const ivec3 &value)
{
    multiSpatial = value;
}

}
}
