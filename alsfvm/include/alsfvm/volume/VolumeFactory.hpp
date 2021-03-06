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
#include "alsfvm/volume/Volume.hpp"

namespace alsfvm {
namespace volume {

///
/// This factory creates volumes for an equation. Both
/// conserved and extra variables.
///
/// Here the idea is that one part of the system creates this factory
/// then passes it along to other parts which needs to create the memory areas.
///
class VolumeFactory {
public:

    ///
    /// Constructs the factory.
    /// \param equation the equation name ("euler1", "euler2", "euler3", "sw", etc.)
    /// \param memoryFactory the memory factory to use
    ///
    VolumeFactory(const std::string& equation,
        alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory);


    ///
    /// Creates a new volume containing the conserved variables.
    /// \param nx the number of cells in x direction
    /// \param ny the number of cells in y direction
    /// \param nz the number of cells in z direction
    /// \param numberOfGhostCells the number of ghostcells to use in each direction
    ///
    alsfvm::shared_ptr<Volume> createConservedVolume(size_t nx, size_t ny,
        size_t nz, size_t numberOfGhostCells = 0);

    ///
    /// Creates a new volume containing the extra variables.
    /// \param nx the number of cells in x direction
    /// \param ny the number of cells in y direction
    /// \param nz the number of cells in z direction
    /// \param numberOfGhostCells the number of ghostcells to use in each direction
    ///
    alsfvm::shared_ptr<Volume> createExtraVolume(size_t nx, size_t ny, size_t nz,
        size_t numberOfGhostCells = 0);

    ///
    /// \brief createPrimitiveVolume creates the primitive volume for the equation,
    /// The primitive volume is often handy for stating initial data for instance.
    /// \param nx the number of cells in x direction
    /// \param ny the number of cells in y direction
    /// \param nz the number of cells in z direction
    /// \param numberOfGhostCells the number of ghostcells to use in each direction
    ///
    alsfvm::shared_ptr<Volume> createPrimitiveVolume(size_t nx, size_t ny,
        size_t nz, size_t numberOfGhostCells = 0);

private:
    std::string equation;
    alsfvm::shared_ptr<alsfvm::memory::MemoryFactory> memoryFactory;
};
} // namespace alsfvm
} // namespace volume
