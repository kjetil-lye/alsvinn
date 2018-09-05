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

#include "alsuq/mpi/LoadBalancerFactory.hpp"
#include "alsuq/mpi/SimpleLoadBalancer.hpp"
#include "alsuq/mpi/MultilevelLoadBalancer.hpp"

namespace alsuq {
namespace mpi {

namespace {
bool isMultilevel(const std::vector<samples::SampleInformation>&
    sampleInformations) {
    for (const auto& sample : sampleInformations) {
        if (sample.getLevel() != 0 || sample.getSign() != 1) {
            return true;
        }
    }

    return false;
}
}

AbstractLoadBalancerPtr LoadBalancerFactory::createLoadBalancer(
    const std::vector<samples::SampleInformation>& sampleInformations) {
    AbstractLoadBalancerPtr pointer;

    if (isMultilevel(sampleInformations)) {
        pointer.reset(new MultilevelLoadBalancer(sampleInformations));
    } else {
        pointer.reset(new SimpleLoadBalancer(sampleInformations));
    }

    return pointer;
}

}
}
