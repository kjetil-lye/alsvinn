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

#include "alsuq/io/MPIWriterFactory.hpp"
#include "alsfvm/io/HDF5MPIWriter.hpp"
#include "alsfvm/io/NetCDFMPIWriter.hpp"
#include "alsfvm/io/PythonScript.hpp"
#include "alsuq/mpi/Configuration.hpp"
namespace alsuq {
namespace io {

MPIWriterFactory::MPIWriterFactory(const std::vector<std::string>& groupNames,
    size_t groupIndex,
    bool createFile,
    MPI_Comm mpiCommunicator, MPI_Info mpiInfo,
    int level, int sign)

    : groupNames(groupNames), groupIndex(groupIndex), createFile(createFile),
      mpiCommunicator(mpiCommunicator),
      mpiInfo(mpiInfo), level(level), sign(sign) {

}

alsfvm::shared_ptr<alsfvm::io::Writer> MPIWriterFactory::createWriter(
    const std::string& name,
    const std::string& baseFilename,
    const alsutils::parameters::Parameters& parameters) {

    auto baseLevelName = baseFilename + std::string("_") + std::to_string(
            level) + std::string("_") + std::to_string(sign > 0);
    mpi::ConfigurationPtr configuration = std::make_shared<mpi::Configuration>
        (mpiCommunicator, "cpu");

    alsfvm::shared_ptr<alsfvm::io::Writer> writer;
    auto parameterCopy = parameters;
    parameterCopy.addIntegerParameter("mpi_rank", configuration->getRank());
    parameterCopy.addIntegerParameter("mpi_size",
        configuration->getNumberOfProcesses());
    parameterCopy.addVectorParameter("group_names", groupNames);
    parameterCopy.addIntegerParameter("group_index", groupIndex);

    if (name == "hdf5") {
        writer.reset(new alsfvm::io::HDF5MPIWriter(baseLevelName, groupNames,
                groupIndex, createFile, mpiCommunicator,
                mpiInfo));
    } else if (name == "netcdf") {
        writer.reset(new alsfvm::io::NetCDFMPIWriter(baseLevelName, groupNames,
                groupIndex, createFile, mpiCommunicator,
                mpiInfo));

    } else if (name == "python") {
        writer.reset(new alsfvm::io::PythonScript(baseLevelName, parameterCopy,
                configuration));
    } else {
        THROW("Unknown writer " << name);
    }

    return writer;
}

}
}
