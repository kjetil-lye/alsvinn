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

#include "alsuq/stats/StatisticsHelper.hpp"
#include "alsuq/mpi/utils.hpp"
#include "alsutils/mpi/cuda.hpp"
#include "alsutils/log.hpp"
#include "alsutils/error/Exception.hpp"

namespace alsuq {
namespace stats {
StatisticsHelper::StatisticsHelper(const StatisticsParameters& parameters)

    : samples(parameters.getNumberOfSamples()),
      mpiConfig(parameters.getMpiConfiguration()),
      exchangeCacheFactory(parameters.getExchangeCacheFactory()) {

}

void StatisticsHelper::addWriter(const std::string& name,
    std::shared_ptr<alsfvm::io::Writer>& writer) {
    writers[name].push_back(writer);
}

void StatisticsHelper::combineStatistics() {

    for (auto& snapshot : snapshots) {
        for (auto& statistics : snapshot.second) {
            for (auto& volume : statistics.second.getVolumes()) {
                for (size_t variable = 0; variable < volume->getNumberOfVariables();
                    variable++) {

                    auto statisticsData = volume->getScalarMemoryArea(variable);
                    auto statisticsDataToReduce = statisticsData;


                    // Check if we should copy to CPU
                    if (!statisticsData->isOnHost() && !alsutils::mpi::hasGPUDirectSupport()) {

                        statisticsDataToReduce = statisticsData->getHostMemory();

                        ALSVINN_LOG(INFO, "Copying from GPU, now statisticsDataToReduce.isOnHost() = "
                            << statisticsDataToReduce->isOnHost() );

                    }


                    std::shared_ptr<alsfvm::memory::Memory<real> > dataReduced;
                    //if (mpiConfig.getRank() == 0) {
                    dataReduced = statisticsDataToReduce->makeInstance();
                    //}
                    MPI_SAFE_CALL(MPI_Reduce(statisticsDataToReduce->data(), dataReduced->data(),
                            statisticsDataToReduce->getSize(), MPI_DOUBLE, MPI_SUM, 0,
                            mpiConfig->getCommunicator()));

                    if (mpiConfig->getRank() == 0) {
                        statisticsData->copyFrom(*dataReduced);
                    }

                    *statisticsData /= samples;
                }
            }
        }
    }
}

void StatisticsHelper::writeStatistics(const alsfvm::grid::Grid& grid) {
    for (auto& snapshot : snapshots) {
        for (auto& statistics : snapshot.second) {
            const auto& statisticsName = statistics.first;

            for (auto& writer : writers[statisticsName]) {
                auto& volumes = statistics.second.getVolumes();
                auto& timestepInformation = statistics.second.getTimestepInformation();
                writer->write(*volumes.getConservedVolume(),
                    *volumes.getExtraVolume(),
                    grid, timestepInformation);
            }
        }
    }
}

void StatisticsHelper::write(const alsfvm::volume::Volume& conservedVariables,
    const alsfvm::volume::Volume& extraVariables, const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {

    Statistics::write(conservedVariables, extraVariables, grid,
        timestepInformation);

    const size_t numberOfGhostCells = getNumberOfGhostCells();

    if (numberOfGhostCells > conservedVariables.getNumberOfXCells()) {
        if (!exchangeCache) {
            exchangeCache = exchangeCacheFactory->makeExchangeCache(grid);
        }

        const int timestep = int(timestepInformation.getNumberOfStepsPerformed());

        for (size_t side = 0; side < 6; ++side) {
            if (exchangeCache->hasSide(side)) {

                exchangeCache->startExchange(conservedVariables, extraVariables,
                    numberOfGhostCells, side, timestep
                );

                auto output = exchangeCache->getBorder(side, timestep);

                this->computeStatisticsBorder(conservedVariables, extraVariables,
                    *output.getConservedVolume(),
                    *output.getExtraVolume(),
                    grid, timestepInformation, side);

            }
        }
    }
}

size_t StatisticsHelper::getNumberOfGhostCells() const {
    return 0;
}

size_t StatisticsHelper::computeStatisticsBorder(const alsfvm::volume::Volume&
    conservedVariablesInner, const alsfvm::volume::Volume&,
    const alsfvm::volume::Volume&,
    const alsfvm::volume::Volume&,
    const alsfvm::grid::Grid&,
    const alsfvm::simulator::TimestepInformation&,
    const size_t) {
    THROW("Calling not implemented computeStatisticsBorder, current state is\n"
        << "\tthis->getNumberOfGhostCells() = " << getNumberOfGhostCells() << "\n,"
        << "\tconservedVariablesInner.getNumberOfGhostCells() = " <<
        conservedVariablesInner.getNumberOfGhostCells() << "\n");
}

StatisticsSnapshot& StatisticsHelper::findOrCreateSnapshot(
    const std::string& name,
    const alsfvm::simulator::TimestepInformation& timestepInformation,
    const alsfvm::volume::Volume& conservedVariables,
    const alsfvm::volume::Volume& extraVariables) {
    auto currentTime = timestepInformation.getCurrentTime();

    if (snapshots.find(currentTime) != snapshots.end()
        && snapshots[currentTime].find(name) != snapshots[currentTime].end()) {
        return snapshots[currentTime][name];
    } else {
        auto conservedVariablesClone = conservedVariables.makeInstance();
        auto extraVariablesClone = extraVariables.makeInstance();
        snapshots[currentTime][name] = StatisticsSnapshot(timestepInformation,
                alsfvm::volume::VolumePair(conservedVariablesClone,
                    extraVariablesClone));

        for (auto& volume : snapshots[currentTime][name].getVolumes()) {
            volume->makeZero();
        }

        return snapshots[currentTime][name];

    }

}

StatisticsSnapshot& StatisticsHelper::findOrCreateSnapshot(
    const std::string& name,
    const alsfvm::simulator::TimestepInformation& timestepInformation,
    const alsfvm::volume::Volume& conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    size_t nx, size_t ny, size_t nz, const std::string& platform) {
    auto currentTime = timestepInformation.getCurrentTime();

    if (snapshots.find(currentTime) != snapshots.end()
        && snapshots[currentTime].find(name) != snapshots[currentTime].end()) {
        return snapshots[currentTime][name];
    } else {
        auto conservedVariablesClone = conservedVariables.makeInstance(nx, ny, nz,
                platform);
        auto extraVariablesClone = extraVariables.makeInstance(nx, ny, nz, platform);
        snapshots[currentTime][name] = StatisticsSnapshot(timestepInformation,
                alsfvm::volume::VolumePair(conservedVariablesClone,
                    extraVariablesClone));

        for (auto& volume : snapshots[currentTime][name].getVolumes()) {
            volume->makeZero();
        }

        return snapshots[currentTime][name];

    }
}



}
}
