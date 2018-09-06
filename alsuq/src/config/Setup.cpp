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

#include "alsuq/config/Setup.hpp"
#include "alsuq/run/FiniteVolumeSimulatorCreator.hpp"
#include "alsuq/generator/GeneratorFactory.hpp"
#include "alsuq/distribution/DistributionFactory.hpp"
#include "alsuq/mpi/SimpleLoadBalancer.hpp"
#include "alsuq/mpi/LoadBalancerFactory.hpp"
#include <boost/property_tree/xml_parser.hpp>
#include <fstream>
#include "alsuq/stats/StatisticsFactory.hpp"
#include "alsfvm/io/WriterFactory.hpp"
#include "alsfvm/io/MpiWriterFactory.hpp"

#include "alsuq/stats/FixedIntervalStatistics.hpp"
#include "alsuq/stats/TimeIntegratedWriter.hpp"
#include "alsuq/samples/SampleInformation.hpp"
#include <boost/algorithm/string.hpp>
#include "alsutils/log.hpp"
#include  <set>
#include "alsutils/timer/Timer.hpp"

namespace alsuq {
namespace config {
// example:
// <samples>1024</samples>
// <generator>auto</generator>
// <parameters>
//   <parameter>
//     <name>a</name>
//     <length>40</length>
//     <type>uniform</type>
//   </parameter>
// </parameters>
// <stats>
//   <stat>
//     meanvar
//   </stat>
// </stats>


std::shared_ptr<run::Runner> Setup::makeRunner(const std::string& inputFilename,
    alsutils::mpi::ConfigurationPtr mpiConfigurationWorld,
    int multiSample, ivec3 multiSpatial) {
    ALSVINN_TIME_BLOCK(alsvinn, uq, init);
    std::ifstream stream(inputFilename);
    ptree configurationBase;
    boost::property_tree::read_xml(stream, configurationBase);
    auto configuration = configurationBase.get_child("config");
    auto sampleStart = readSampleStart(configuration);

    ALSVINN_LOG(INFO, "sampleStart = " << sampleStart);
    auto numberOfSamplesPerLevel = readNumberOfSamples(configuration);


    auto numberOfSamplesPerLevelPerSign = makeNumberOfSamplesPerLevelPerSign(
            numberOfSamplesPerLevel);

    auto sampleInformations = makeSamplesVector(numberOfSamplesPerLevel,
            sampleStart);

    int numberOfUniqueSamples = getNumberOfUniqueSamples(sampleInformations);

    auto sampleGenerator = makeSampleGenerator(configuration,
            numberOfUniqueSamples);

    mpi::AbstractLoadBalancerPtr loadBalancer =
        mpi::LoadBalancerFactory::createLoadBalancer(sampleInformations);


    auto loadBalanceTuple = loadBalancer->loadBalance(multiSample,
            multiSpatial,
            *mpiConfigurationWorld);

    auto samplesForProc = std::get<1>(loadBalanceTuple);
    auto levelConfiguration = std::get<0>(loadBalanceTuple);


    auto simulatorCreator = std::dynamic_pointer_cast<run::SimulatorCreator>
        (std::make_shared<run::FiniteVolumeSimulatorCreator>
            (inputFilename,
                mpiConfigurationWorld
            ));

    auto name = boost::algorithm::trim_copy(
            configuration.get<std::string>("fvm.name"));
    auto runner = std::make_shared<run::Runner>(simulatorCreator, sampleGenerator,
            samplesForProc, levelConfiguration, name);
    auto statistics  = createStatistics(configuration, mpiConfigurationWorld,
            levelConfiguration,
            numberOfSamplesPerLevelPerSign);
    runner->setStatistics(statistics);

    // We want to make sure everything is created before going further
    MPI_Barrier(mpiConfigurationWorld->getCommunicator());
    return runner;
}

std::map<int, std::map<int, int> > Setup::makeNumberOfSamplesPerLevelPerSign(
    const std::vector<size_t>& numberOfSamplesPerLevel) {

    std::map<int, std::map<int, int> > numberOfSamplesPerLevelPerSign;

    for (size_t level = 0; level < numberOfSamplesPerLevel.size(); ++level) {
        numberOfSamplesPerLevelPerSign[level][1] = numberOfSamplesPerLevel[level];

        if (level < numberOfSamplesPerLevel.size() - 1) {
            numberOfSamplesPerLevelPerSign[level + 1][-1] = numberOfSamplesPerLevel[level];

        }
    }

    return numberOfSamplesPerLevelPerSign;
}

std::shared_ptr<samples::SampleGenerator> Setup::makeSampleGenerator(
    Setup::ptree& configuration, int numberOfSamples) {


    samples::SampleGenerator::GeneratorDistributionMap generators;

    auto generatorName = configuration.get<std::string>("uq.generator");

    if (generatorName == "auto") {
        generatorName = "stlmersenne";
    }

    auto parametersNode = configuration.get_child("uq.parameters");
    generator::GeneratorFactory generatorFactory;
    distribution::DistributionFactory distributionFactory;

    for (auto parameterNode : parametersNode) {
        auto name = parameterNode.second.get<std::string>("name");
        auto length = parameterNode.second.get<size_t>("length");
        auto type = parameterNode.second.get<std::string>("type");

        distribution::Parameters parametersToDistribution(parameterNode.second);
        parametersToDistribution.setParameter("lower", 0);
        parametersToDistribution.setParameter("upper", 1);

        parametersToDistribution.setParameter("a", 0);
        parametersToDistribution.setParameter("b", 1);


        parametersToDistribution.setParameter("mean", 0);
        parametersToDistribution.setParameter("sd", 1);

        auto distribution = distributionFactory.createDistribution(type,
                length,
                numberOfSamples,
                parametersToDistribution);


        auto generator = generatorFactory.makeGenerator(generatorName, length,
                numberOfSamples);
        generators[name] = std::make_pair(length, std::make_pair(generator,
                    distribution));
    }



    return std::make_shared<samples::SampleGenerator> (generators);


}


// <stats>
//
//   <stat>
//   <name>
//   structure
//   </name>
//   <numberOfSaves>1</numberOfSaves>
//   <writer>
//
//     <type>netcdf</type>
//     <basename>kh_structure</basename>
//   </writer>
//   </stat>
//   <stat>
//   <name>
//     meanvar
//   </name>
//   <numberOfSaves>1</numberOfSaves>
//   <writer>
//     <type>netcdf</type>
//     <basename>kh_structure</basename>
//   </writer>
//   </stat>
// </stats>
std::map < int, std::map<int, std::vector<std::shared_ptr<stats::Statistics> > > > Setup::createStatistics(
    ptree& configuration,
    mpi::ConfigurationPtr worldConfiguration,
    mpi::LevelConfiguration levelConfiguration,
    const std::map<int, std::map<int, int> >& samplesPerLevelSign) {
    auto statisticsNodes = configuration.get_child("uq.stats");
    stats::StatisticsFactory statisticsFactory;


    std::map < int, std::map<int, std::vector<std::shared_ptr<stats::Statistics> > > >
    statisticsMap;

    for (int level = 0; level < levelConfiguration.getNumberOfLevels(); ++level) {
        for (int sign : levelConfiguration.getSigns(level)) {

            auto numberOfSamplesAtLevel = samplesPerLevelSign.at(level).at(sign);
            std::shared_ptr<alsfvm::io::WriterFactory> writerFactory;


            auto spatialConfiguration = levelConfiguration.getSpatialConfiguration(level,
                    sign);

            if (spatialConfiguration->getNumberOfProcesses() > 1) {
                writerFactory.reset(new alsfvm::io::MpiWriterFactory(spatialConfiguration));
            } else {
                writerFactory.reset(new alsfvm::io::WriterFactory());
            }

            auto platform = configuration.get<std::string>("fvm.platform");
            std::vector<std::shared_ptr<stats::Statistics> > statisticsVector;

            for (auto& statisticsNode : statisticsNodes) {
                auto name = statisticsNode.second.get<std::string>("name");
                boost::trim(name);
                stats::StatisticsParameters parameters(statisticsNode.second);

                parameters.setNumberOfSamples(numberOfSamplesAtLevel);
                parameters.setPlatform(platform);
                auto statistics = statisticsFactory.makeStatistics(platform, name, parameters);




                // Make writer
                std::string type = statisticsNode.second.get<std::string>("writer.type");
                std::string basename =
                    statisticsNode.second.get<std::string>("writer.basename");

                basename = basename + "_" + std::to_string(level) + "_" + std::to_string(
                        sign > 0);

                statistics->setMpiSpatialConfiguration(
                    levelConfiguration.getSpatialConfiguration(level, sign));
                statistics->setMpiStochasticConfiguration(
                    levelConfiguration.getStochasticConfiguration(level, sign));

                for (auto statisticsName : statistics->getStatisticsNames()) {

                    auto outputname = basename + "_" + statisticsName;
                    auto baseWriter = writerFactory->createWriter(type, outputname,
                            alsfvm::io::Parameters(statisticsNode.second.get_child("writer")));
                    baseWriter->addAttributes("uqAttributes", configuration);
                    statistics->addWriter(statisticsName, baseWriter);
                }

                if (statisticsNode.second.find("numberOfSaves") !=
                    statisticsNode.second.not_found()) {



                    auto numberOfSaves = statisticsNode.second.get<size_t>("numberOfSaves");
                    ALSVINN_LOG(INFO, "statistics.numberOfSaves = " << numberOfSaves);
                    real endTime = configuration.get<real>("fvm.endTime");
                    real timeInterval = endTime / numberOfSaves;
                    auto statisticsInterval =
                        std::shared_ptr<stats::Statistics>(
                            new stats::FixedIntervalStatistics(statistics, timeInterval,
                                endTime));
                    statisticsVector.push_back(statisticsInterval);
                } else if (statisticsNode.second.find("time") !=
                    statisticsNode.second.not_found()) {
                    const real time = statisticsNode.second.get<real>("time");
                    const real radius = statisticsNode.second.get<real>("timeRadius");

                    auto timeIntegrator = std::shared_ptr<stats::Statistics>(
                            new stats::TimeIntegratedWriter(statistics, time,
                                radius));
                    statisticsVector.push_back(timeIntegrator);
                }


            }

            statisticsMap[level][sign] = statisticsVector;
        }
    }

    return statisticsMap;
}

std::vector<size_t> Setup::readNumberOfSamples(Setup::ptree& configuration) {
    auto samplesText =  configuration.get<std::string>("uq.samples");

    std::vector<std::string> samplesPerLevelText;
    boost::split(samplesPerLevelText, samplesText, boost::is_any_of(" \t"));

    std::vector<size_t> samplesPerLevel;

    for (const auto& samplesText : samplesPerLevelText) {
        samplesPerLevel.push_back(boost::lexical_cast<size_t>(samplesText));
    }

    return samplesPerLevel;

}

std::vector<samples::SampleInformation> Setup::makeSamplesVector(
    const std::vector<size_t>& samplesPerLevel, size_t sampleStart) {
    std::vector<samples::SampleInformation> sampleInformation;

    for (size_t level = 0; level < samplesPerLevel.size(); ++level) {
        for (size_t sample = 0; sample < samplesPerLevel[level]; ++sample) {
            sampleInformation.push_back(samples::SampleInformation(sampleStart + sample,
                    level, 1));

            if (level < samplesPerLevel.size() - 1) {
                sampleInformation.push_back(samples::SampleInformation(sampleStart + sample,
                        level + 1, -1));
            }
        }
    }

    return sampleInformation;
}

int Setup::getNumberOfUniqueSamples(const
    std::vector<samples::SampleInformation>& samples) {

    ALSVINN_TIME_BLOCK(alsvinn, uq, find_unique_samples);
    std::set<int> sampleIds;

    for (const auto& sample : samples) {
        sampleIds.insert(sample.getSampleNumber());
    }

    return sampleIds.size();
}

size_t Setup::readSampleStart(Setup::ptree& configuration) {
    auto& uq = configuration.get_child("uq");

    if (uq.find("sampleStart") != uq.not_found()) {
        ALSVINN_LOG(INFO, "sampleStart tag present");
        return uq.get<size_t>("sampleStart");
    }

    return 0;
}
}
}
