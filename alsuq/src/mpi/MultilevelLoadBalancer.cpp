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

#include <map>
#include "alsuq/mpi/MultilevelLoadBalancer.hpp"
#include "alsutils/error/Exception.hpp"

namespace alsuq {
namespace mpi {

MultilevelLoadBalancer::MultilevelLoadBalancer(const
    std::vector<samples::SampleInformation>& samples)
    : samples(samples) {

}

std::tuple
< LevelConfiguration, std::vector<samples::SampleInformation> > MultilevelLoadBalancer::loadBalance(
    int multiSample, ivec3 multiSpatial,
    const alsutils::mpi::Configuration& mpiConfig) {
    // Before going through this method, read a basic tutorial on mpi communicators,
    // eg http://mpitutorial.com/tutorials/introduction-to-groups-and-communicators/


    int totalNumberOfProcesses = mpiConfig.getNumberOfProcesses();

    if (multiSample * multiSpatial.x * multiSpatial.y  * multiSpatial.z !=
        int(totalNumberOfProcesses)) {
        THROW("The number of processors given (" << totalNumberOfProcesses
            << ") does not match the distribution of the samples / spatial dimensions. We were given:\n"
            << "\tmultiSample: " << multiSample << "\n"
            << "\tmultiSpatial: " << multiSpatial << "\n\n"
            << "We require that\n"
            "\tmultiSample * multiSpatial.x * multiSpatial.y  * multiSpatial.z == totalNumberOfProcesses");
    }



    std::map<int, std::vector<samples::SampleInformation> > samplesPerLevel;

    int maximumLevel = 0;

    for (auto& sampleInformation : samples) {
        const int level = sampleInformation.getLevel();
        samplesPerLevel[level].push_back(sampleInformation);

        maximumLevel = std::max(level, maximumLevel);
    }

    const int globalRank = mpiConfig.getRank();
    std::vector<samples::SampleInformation> samplesForProcess;

    LevelConfiguration levelConfiguration;

    for (int level = 0; level <= maximumLevel; ++level) {
        const int totalNumberOfSamples = int(samplesPerLevel[level].size());

        if (totalNumberOfSamples % multiSample != 0) {
            THROW("The number of processors must be a divisor of the numberOfSamples.\n"
                << "ie. totalNumberOfSamples = N * numberOfProcesses     for some integer N"
                << "\n\n"
                << "We were given:"
                << "\n\ttotalNumberOfSamples = " << totalNumberOfSamples
                << "\n\tmultiSample = " << multiSample
                << "\n\nThis can be changed in the config file by editing"
                << "\n\t<samples>NUMBER OF SAMPLES</samples>"
                << "\n\n"
                << "and as a parameter to mpirun (-np) eg."
                << "\n\tmpirun -np 48 alsuq <path to xml>"
                << "\n\n\nNOTE:This is a strict requirement, otherwise we have to start"
                << "\nreconfiguring the communicator when the last process runs out of samples.");
        }

        int numberOfSamplesPerProcess = (totalNumberOfSamples) / multiSample;


        int numberOfProcessorsPerSample = multiSpatial.x * multiSpatial.y *
            multiSpatial.z;

        const int statisticalRank = globalRank / numberOfProcessorsPerSample;
        const int spatialRank = globalRank % numberOfProcessorsPerSample;

        auto statisticalConfiguration = mpiConfig.makeSubConfiguration(spatialRank,
                statisticalRank);
        auto spatialConfiguration = mpiConfig.makeSubConfiguration(statisticalRank,
                spatialRank);

        int rank = statisticalConfiguration->getRank();



        for (int i = numberOfSamplesPerProcess * rank;
            i < numberOfSamplesPerProcess * (rank + 1);
            ++i) {

            if (i >= totalNumberOfSamples) {
                THROW("Something went wrong in load balancing."
                    << "\nThe parameters were\n"
                    << "\n\ttotalNumberOfSamples = " << totalNumberOfSamples
                    << "\n\tnumberOfProcesses = " << totalNumberOfProcesses
                    << "\n\tnumberOfSamplesPerProcess = " << numberOfSamplesPerProcess
                    << "\n\trank = " << rank
                    << "\n\ti= " << i);
            }

            auto sample = samplesPerLevel[level][size_t(i)];
            sample.setSpatialMpiConfiguration(spatialConfiguration);
            sample.setStochasticMpiConfiguration(
                statisticalConfiguration);
            sample.setMultiSpatial(multiSpatial);

            levelConfiguration.setSpatialConfiguration(level, sample.getSign(),
                spatialConfiguration);
            levelConfiguration.setStochasticConfiguration(level, sample.getSign(),
                statisticalConfiguration);

            samplesForProcess.push_back(sample);
        }

        for (size_t i = 0; i < 3u; ++i) {
            multiSpatial[i] = std::max(1, multiSpatial[i] / 2);
        }

        multiSample = std::max(1, multiSample / 2);
    }

    return std::make_tuple
        (levelConfiguration, samplesForProcess);
}

}
}
