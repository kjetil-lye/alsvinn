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
#include "alsuq/stats/StatisticsHelper.hpp"
#include "alsuq/types.hpp"
namespace alsuq {
namespace stats {

//! Computes the sturcture function given a direction
//!
//! Ie in parameters it gets an in direction that corresponds to the unit
//! direction vectors \f$e\f$
//!
//! It then computes the structure function as
//!
//! \f[\sum_{i,j,k} |u_{(i,j,k) + e}-u_{(i,j,k)}|^p\f]
class StructureBasic : public StatisticsHelper {
public:
    StructureBasic(const StatisticsParameters& parameters);


    //! Returns a list of ['structure_basic']
    virtual std::vector<std::string> getStatisticsNames() const override;




    virtual void computeStatistics(const alsfvm::volume::Volume& conservedVariables,
        const alsfvm::volume::Volume& extraVariables,
        const alsfvm::grid::Grid& grid,
        const alsfvm::simulator::TimestepInformation& timestepInformation) override;

    virtual void finalizeStatistics() override;

protected:
    virtual size_t getNumberOfGhostCells() const override;
    virtual size_t computeStatisticsBorder(const alsfvm::volume::Volume&
        conservedVariablesInner,
        const alsfvm::volume::Volume& extraVariablesInner,
        const alsfvm::volume::Volume& conservedVariablesBorder,
        const alsfvm::volume::Volume& extraVariablesBorder,
        const alsfvm::grid::Grid& grid,
        const alsfvm::simulator::TimestepInformation& timestepInformation,
        const size_t side) override;


private:
    void computeStructure(alsfvm::volume::Volume& outputVolume,
        const alsfvm::volume::Volume& input);

    const size_t direction;
    const real p;
    const ivec3 directionVector;
    const size_t numberOfH;
    const std::string statisticsName;

};
} // namespace stats
} // namespace alsuq
