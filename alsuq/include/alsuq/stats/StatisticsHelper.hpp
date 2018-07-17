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
#include "alsuq/types.hpp"
#include "alsuq/stats/Statistics.hpp"
#include "alsuq/stats/StatisticsSnapshot.hpp"
#include "alsuq/stats/StatisticsParameters.hpp"

namespace alsuq {
namespace stats {

class StatisticsHelper : public Statistics {
public:

    StatisticsHelper(const StatisticsParameters& parameters);


    //! Add a writer to write the statistics to file
    //!
    //! @param writer the writer to add
    //! @param name the name of the statistics (must be unique)
    void addWriter(const std::string& name,
        std::shared_ptr<alsfvm::io::Writer>& writer) override;

    //! Should be called at the end of the simulation
    virtual void combineStatistics() override;



    //! Writes the statistics to file
    virtual void writeStatistics(const alsfvm::grid::Grid& grid) override;



protected:
    std::map<real, std::map<std::string, StatisticsSnapshot> > snapshots;

    //! Utility function.
    //!
    //! If the given timstep is already created, return that timestep,
    //! otherwise creates a new snapshot
    //!
    //! \note Uses the size of the given volume
    StatisticsSnapshot& findOrCreateSnapshot(const std::string& name,
        const alsfvm::simulator::TimestepInformation& timestepInformation,
        const alsfvm::volume::Volume& conservedVariables,
        const alsfvm::volume::Volume& extraVariables);

    //! Utility function.
    //!
    //! If the given timstep is already created, return that timestep,
    //! otherwise creates a new snapshot
    StatisticsSnapshot& findOrCreateSnapshot(const std::string& name,
        const alsfvm::simulator::TimestepInformation& timestepInformation,
        const alsfvm::volume::Volume& conservedVariables,
        const alsfvm::volume::Volume& extraVariables,
        size_t nx, size_t ny, size_t nz, const std::string& platform = "default");

    //! Gets the data on the border, if any.
    //!
    //! @note returns an empty volume if the border is not present (ie. if this node is at the end)
    //!
    //! @note This is not the most efficient way of doing this,
    //!       but since we anyway don't expect the statistics to be called too often,
    //!       it should be ok.
    alsfvm::volume::VolumePair getBorder(ivec3 sideIdentifcation, int ghostSize);
private:
    size_t samples;

    std::map<std::string, std::vector<std::shared_ptr<alsfvm::io::Writer>  > >
    writers;

    alsuq::mpi::ConfigurationPtr mpiConfig;
};
} // namespace stats
} // namespace alsuq
