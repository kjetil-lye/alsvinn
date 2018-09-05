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

#include "alsuq/run/FiniteVolumeSimulatorCreator.hpp"
#include "alsfvm/config/SimulatorSetup.hpp"
#include "alsuq/io/MPIWriterFactory.hpp"
#include "alsuq/mpi/Configuration.hpp"
#include "alsuq/mpi/utils.hpp"

namespace alsuq {
namespace run {

FiniteVolumeSimulatorCreator::FiniteVolumeSimulatorCreator(
    const std::string& configurationFile,
    alsutils::mpi::ConfigurationPtr mpiConfigurationWorld)
    :
    mpiConfigurationWorld(mpiConfigurationWorld),

    filename(configurationFile) {

}

alsfvm::shared_ptr<alsfvm::simulator::AbstractSimulator> FiniteVolumeSimulatorCreator::createSimulator(
    const alsfvm::init::Parameters& initialDataParameters,
    const alsuq::samples::SampleInformation& sample) {
    const int level = sample.getLevel();

    const int sign = sample.getSign();
    auto mpiConfigurationSpatial = sample.getSpatialMpiConfiguration();
    auto mpiConfigurationStatistical = sample.getStochasticMpiConfiguration();

    auto multiSpatial = sample.getMultiSpatial();
    auto groupNames = makeGroupNames(sample);
    std::shared_ptr<alsfvm::io::WriterFactory> writerFactory(
        new io::MPIWriterFactory(groupNames, mpiConfigurationStatistical->getRank(),
            !notFirstCall[level][sign], mpiConfigurationWorld->getCommunicator(),
            mpiConfigurationWorld->getInfo(), level, sign));

    notFirstCall[level][sign] = true;
    alsfvm::config::SimulatorSetup simulatorSetup;

    simulatorSetup.enableMPI(mpiConfigurationSpatial, multiSpatial.x,
        multiSpatial.y,
        multiSpatial.z);

    simulatorSetup.setNumberOfGridCellsScaling(1 << sample.getLevel());
    simulatorSetup.setWriterFactory(writerFactory);
    auto simulatorPair = simulatorSetup.readSetupFromFile(filename);

    auto simulator = simulatorPair.first;
    auto initialData = simulatorPair.second;

    initialData->setParameters(initialDataParameters);

    simulator->setInitialValue(initialData);
    return std::dynamic_pointer_cast<alsfvm::simulator::AbstractSimulator>
        (simulator);
}

std::vector<std::string> FiniteVolumeSimulatorCreator::makeGroupNames(
    const samples::SampleInformation& sample) {
    auto mpiConfigurationStatistical = sample.getStochasticMpiConfiguration();
    std::vector<int> samples(
        size_t(mpiConfigurationStatistical->getNumberOfProcesses()));

    int sampleNumber = sample.getSampleNumber();
    MPI_SAFE_CALL(MPI_Allgather(static_cast<void*>(&sampleNumber), 1, MPI_INT,
            static_cast<void*>( samples.data()), 1,
            MPI_INT, mpiConfigurationStatistical->getCommunicator()));




    std::vector<std::string> groupNames;
    groupNames.reserve(samples.size());

    for (size_t sampleNumber : samples) {
        groupNames.push_back("sample_" + sampleNumber);
    }

    return groupNames;
}

}
}
