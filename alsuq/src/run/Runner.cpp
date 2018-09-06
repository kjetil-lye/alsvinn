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

#include "alsuq/run/Runner.hpp"
#include "alsuq/mpi/LevelConfiguration.hpp"

#include "alsutils/log.hpp"

namespace alsuq {
namespace run {

Runner::Runner(std::shared_ptr<SimulatorCreator> simulatorCreator,
    std::shared_ptr<samples::SampleGenerator> sampleGenerator,
    const std::vector<samples::SampleInformation>& samples,
    alsuq::mpi::LevelConfiguration mpiConfig,
    const std::string& name)
    : simulatorCreator(simulatorCreator),
      sampleGenerator(sampleGenerator),
      parameterNames(sampleGenerator->getParameterList()),
      samples(samples),
      mpiConfig(mpiConfig),
      name(name)

{

}

void Runner::run() {
    std::map<int, std::map<int, std::shared_ptr<alsfvm::grid::Grid> > > grid;

    for (auto& sample : samples) {
        ALSVINN_LOG(INFO, "Running sample: " << sample.getSampleNumber() << std::endl);
        alsfvm::init::Parameters parameters;

        for (auto parameterName : parameterNames) {
            auto parametersValues = sampleGenerator->generate(parameterName,
                    sample.getSampleNumber());
            parameters.addParameter(parameterName,
                parametersValues);

        }

        auto simulator = simulatorCreator->createSimulator(parameters, sample);


        for ( auto& statisticWriter : statistics.at(sample.getLevel()).at(
                sample.getSign())) {


            simulator->addWriter(std::dynamic_pointer_cast<alsfvm::io::Writer>
                (statisticWriter));

            auto timestepAdjuster =
                alsfvm::dynamic_pointer_cast<alsfvm::integrator::TimestepAdjuster>
                (statisticWriter);

            if (timestepAdjuster) {
                simulator->addTimestepAdjuster(timestepAdjuster);
            }
        }

        simulator->callWriters();

        while (!simulator->atEnd()) {
            simulator->performStep();
            timestepsPerformedTotal++;
        }

        simulator->finalize();
        grid[sample.getLevel()][sample.getSign()] = simulator->getGrid();

    }

    for (auto& level : statistics) {
        for (auto& sign : level.second) {
            for (auto& statisticsWriter : sign.second) {
                statisticsWriter->combineStatistics();

                if (mpiConfig.getStochasticConfiguration(level.first,
                        sign.first)->getRank() == 0) {
                    statisticsWriter->finalizeStatistics();
                    statisticsWriter->writeStatistics(*grid[level.first][sign.first]);
                }
            }
        }
    }


}


void Runner::setStatistics(const
    std::map<int, std::map<int, std::vector<std::shared_ptr<stats::Statistics> > > >
    &
    statistics) {
    this->statistics = statistics;
}

std::string Runner::getName() const {
    return name;
}

size_t Runner::getTimestepsPerformedTotal() const {
    return timestepsPerformedTotal;
}

}
}
