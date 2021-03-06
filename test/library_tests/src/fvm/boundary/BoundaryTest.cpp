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

#include <gtest/gtest.h>
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/boundary/BoundaryFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/euler/AllVariables.hpp"

using namespace alsfvm;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm::boundary;

struct BoundaryTest : public ::testing::Test {
    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
    alsfvm::shared_ptr<MemoryFactory> memoryFactory;
    std::string equation = "euler3";
    alsfvm::shared_ptr<VolumeFactory> volumeFactory;
    size_t nx = 10, ny = 11, nz = 12;
    size_t ghostCells = 2;
    rvec3 lowerCorner = rvec3(0, 0, 0);
    rvec3 upperCorner = rvec3(1, 1, 1);
    ivec3 dimensions = ivec3(nx, ny, nz);

    grid::Grid grid;
    alsfvm::shared_ptr<BoundaryFactory> boundaryFactory;
    BoundaryTest()
        : deviceConfiguration(new DeviceConfiguration("cpu")),
          memoryFactory(new MemoryFactory(deviceConfiguration)),
          volumeFactory(new VolumeFactory(equation, memoryFactory)),
          grid(lowerCorner, upperCorner, dimensions) {

    }


};


TEST_F(BoundaryTest, NeumannTest2CellsConstant) {
    boundaryFactory.reset(new BoundaryFactory("neumann", deviceConfiguration));

    auto volume = volumeFactory->createConservedVolume(nx, ny, nz, ghostCells);
    volume->makeZero();
    const real C = 10;
    fill_volume<equation::euler::ConservedVariables<3>>(*volume, grid, [&](real x,
    real y, real z, equation::euler::ConservedVariables<3>& out) {
        out.E = C;
        out.m.x = C;
        out.m.y = C;
        out.m.z = C;
        out.rho = C;
    });

    auto boundary = boundaryFactory->createBoundary(ghostCells);

    boundary->applyBoundaryConditions(*volume, grid);


    for_each_cell_index(*volume, [&](size_t index) {
        ASSERT_EQ(volume->getScalarMemoryArea("rho")->getPointer()[index], C);
    });
}


TEST_F(BoundaryTest, NeumannTest2CellsVarying) {
    boundaryFactory.reset(new BoundaryFactory("neumann", deviceConfiguration));

    auto volume = volumeFactory->createConservedVolume(nx, ny, nz, ghostCells);
    volume->makeZero();

    auto rho = volume->getScalarMemoryArea("rho")->getView();
    auto mx = volume->getScalarMemoryArea("mx")->getView();
    auto my = volume->getScalarMemoryArea("my")->getView();
    auto mz = volume->getScalarMemoryArea("mz")->getView();
    auto E = volume->getScalarMemoryArea("E")->getView();

    for (size_t x = ghostCells; x < nx + ghostCells;
        ++x) { // NOTE: The endpoint is correct, this is indexing wrt. full volume, WITH ghost cells
        for (size_t y = ghostCells; y < ny + ghostCells;
            ++y) { // NOTE: The endpoint is correct, this is indexing wrt. full volume, WITH ghost cells
            for (size_t z = ghostCells; z < nz + ghostCells;
                ++z) { // NOTE: The endpoint is correct, this is indexing wrt. full volume, WITH ghost cells
                rho.at(x, y, z) = rho.index(x, y, z);
                mx.at(x, y, z) = rho.index(x, y, z);
                my.at(x, y, z) = rho.index(x, y, z);
                mz.at(x, y, z) = rho.index(x, y, z);
                E.at(x, y, z) = rho.index(x, y, z);
            }
        }
    }


    auto boundary = boundaryFactory->createBoundary(ghostCells);

    boundary->applyBoundaryConditions(*volume, grid);

    // X side
    for (size_t y = ghostCells; y < ny + ghostCells; ++y) {
        for (size_t z = ghostCells; z < nz + ghostCells; ++z) {
            ASSERT_EQ(rho.at(0, y, z), rho.index(3, y, z));
            ASSERT_EQ(rho.at(1, y, z), rho.index(2, y, z));

            ASSERT_EQ(rho.at(ghostCells + nx + 1, y, z), rho.index(nx + ghostCells - 2, y,
                    z))
                    << "Wrong index at (" << (ghostCells + nx + 1) << ", " << y << ", " << z << ")";
            ASSERT_EQ(rho.at(ghostCells + nx, y, z), rho.index(nx + ghostCells - 1, y, z))
                    << "Wrong index at (" << (ghostCells + nx) << ", " << y << ", " << z << ")";;
        }
    }

    // Y side
    for (size_t x = ghostCells; x < nx + ghostCells; ++x) {
        for (size_t z = ghostCells; z < nz + ghostCells; ++z) {
            ASSERT_EQ(rho.at(x, 0, z), rho.index(x, 3, z));
            ASSERT_EQ(rho.at(x, 1, z), rho.index(x, 2, z));

            ASSERT_EQ(rho.at(x, ghostCells + ny + 1, z), rho.index(x, ny + ghostCells - 2,
                    z));
            ASSERT_EQ(rho.at(x, ghostCells + ny, z), rho.index(x, ny + ghostCells - 1, z));
        }
    }

    // Z side
    for (size_t x = ghostCells; x < nx + ghostCells; ++x) {
        for (size_t y = ghostCells; y < ny + ghostCells; ++y) {
            ASSERT_EQ(rho.at(x, y, 0), rho.index(x, y, 3));
            ASSERT_EQ(rho.at(x, y, 1), rho.index(x, y, 2));

            ASSERT_EQ(rho.at(x, y, ghostCells + nz + 1), rho.index(x, y,
                    nz + ghostCells - 2));
            ASSERT_EQ(rho.at(x, y, ghostCells + nz), rho.index(x, y, nz + ghostCells - 1));
        }
    }
}



TEST_F(BoundaryTest, NeumannTest2CellsVarying2D) {
    boundaryFactory.reset(new BoundaryFactory("neumann", deviceConfiguration));

    auto volume = volumeFactory->createConservedVolume(nx, ny, 1, ghostCells);
    volume->makeZero();

    auto rho = volume->getScalarMemoryArea("rho")->getView();
    auto mx = volume->getScalarMemoryArea("mx")->getView();
    auto my = volume->getScalarMemoryArea("my")->getView();
    auto mz = volume->getScalarMemoryArea("mz")->getView();
    auto E = volume->getScalarMemoryArea("E")->getView();

    size_t z = 0;

    for (size_t x = ghostCells; x < nx + ghostCells; ++x) {
        for (size_t y = ghostCells; y < ny + ghostCells; ++y) {

            rho.at(x, y, z) = rho.index(x, y, z);
            mx.at(x, y, z) = rho.index(x, y, z);
            my.at(x, y, z) = rho.index(x, y, z);
            mz.at(x, y, z) = rho.index(x, y, z);
            E.at(x, y, z) = rho.index(x, y, z);

        }
    }


    auto boundary = boundaryFactory->createBoundary(ghostCells);

    auto smallerGrid = alsfvm::make_shared<grid::Grid>(rvec3(0, 0, 0), rvec3(1, 1,
                0), ivec3(nx, ny, 1));
    boundary->applyBoundaryConditions(*volume, *smallerGrid);

    // X side
    for (size_t y = ghostCells; y < ny + ghostCells; ++y) {
        ASSERT_EQ(rho.at(0, y, z), rho.index(3, y, z));
        ASSERT_EQ(rho.at(1, y, z), rho.index(2, y, z));

        ASSERT_EQ(rho.at(ghostCells + nx + 1, y, z), rho.index(nx + ghostCells - 2, y,
                z));
        ASSERT_EQ(rho.at(ghostCells + nx, y, z), rho.index(nx + ghostCells - 1, y, z));
    }
}



TEST_F(BoundaryTest, PeriodicTest2CellsConstant) {
    boundaryFactory.reset(new BoundaryFactory("periodic", deviceConfiguration));

    auto volume = volumeFactory->createConservedVolume(nx, ny, nz, ghostCells);
    volume->makeZero();
    const real C = 10;
    fill_volume<equation::euler::ConservedVariables<3>>(*volume, grid, [&](real x,
    real y, real z, equation::euler::ConservedVariables<3>& out) {
        out.E = C;
        out.m.x = C;
        out.m.y = C;
        out.m.z = C;
        out.rho = C;
    });

    auto boundary = boundaryFactory->createBoundary(ghostCells);

    boundary->applyBoundaryConditions(*volume, grid);


    for_each_cell_index(*volume, [&](size_t index) {
        ASSERT_EQ(volume->getScalarMemoryArea("rho")->getPointer()[index], C);
    });
}


TEST_F(BoundaryTest, PeriodicTest2CellsVarying) {
    boundaryFactory.reset(new BoundaryFactory("periodic", deviceConfiguration));

    auto volume = volumeFactory->createConservedVolume(nx, ny, nz, ghostCells);
    volume->makeZero();

    auto rho = volume->getScalarMemoryArea("rho")->getView();
    auto mx = volume->getScalarMemoryArea("mx")->getView();
    auto my = volume->getScalarMemoryArea("my")->getView();
    auto mz = volume->getScalarMemoryArea("mz")->getView();
    auto E = volume->getScalarMemoryArea("E")->getView();

    for (size_t x = ghostCells; x < nx + ghostCells;
        ++x) { // NOTE: The endpoint is correct, this is indexing wrt. full volume, WITH ghost cells
        for (size_t y = ghostCells; y < ny + ghostCells;
            ++y) { // NOTE: The endpoint is correct, this is indexing wrt. full volume, WITH ghost cells
            for (size_t z = ghostCells; z < nz + ghostCells;
                ++z) { // NOTE: The endpoint is correct, this is indexing wrt. full volume, WITH ghost cells
                rho.at(x, y, z) = rho.index(x, y, z);
                mx.at(x, y, z) = rho.index(x, y, z);
                my.at(x, y, z) = rho.index(x, y, z);
                mz.at(x, y, z) = rho.index(x, y, z);
                E.at(x, y, z) = rho.index(x, y, z);
            }
        }
    }


    auto boundary = boundaryFactory->createBoundary(ghostCells);

    boundary->applyBoundaryConditions(*volume, grid);

    // X side
    for (size_t y = ghostCells; y < ny + ghostCells; ++y) {
        for (size_t z = ghostCells; z < nz + ghostCells; ++z) {
            ASSERT_EQ(rho.at(0, y, z), rho.index(nx - 2 + ghostCells, y, z));
            ASSERT_EQ(rho.at(1, y, z), rho.index(nx - 1 + ghostCells, y, z));

            ASSERT_EQ(rho.at(ghostCells + nx + 1, y, z), rho.index(3, y, z))
                    << "Wrong index at (" << (ghostCells + nx + 1) << ", " << y << ", " << z << ")";
            ASSERT_EQ(rho.at(ghostCells + nx, y, z), rho.index(2, y, z))
                    << "Wrong index at (" << (ghostCells + nx) << ", " << y << ", " << z << ")";;
        }
    }

    // Y side
    for (size_t x = ghostCells; x < nx + ghostCells; ++x) {
        for (size_t z = ghostCells; z < nz + ghostCells; ++z) {
            ASSERT_EQ(rho.at(x, 0, z), rho.index(x, ny - 2 + ghostCells, z));
            ASSERT_EQ(rho.at(x, 1, z), rho.index(x, ny - 1 + ghostCells, z));

            ASSERT_EQ(rho.at(x, ghostCells + ny + 1, z), rho.index(x, 3, z));
            ASSERT_EQ(rho.at(x, ghostCells + ny, z), rho.index(x, 2, z));
        }
    }

    // Z side
    for (size_t x = ghostCells; x < nx + ghostCells; ++x) {
        for (size_t y = ghostCells; y < ny + ghostCells; ++y) {
            ASSERT_EQ(rho.at(x, y, 0), rho.index(x, y, nz - 2 + ghostCells));
            ASSERT_EQ(rho.at(x, y, 1), rho.index(x, y, nz - 1 + ghostCells));

            ASSERT_EQ(rho.at(x, y, ghostCells + nz + 1), rho.index(x, y, 3));
            ASSERT_EQ(rho.at(x, y, ghostCells + nz), rho.index(x, y, 2));
        }
    }
}



TEST_F(BoundaryTest, PeriodicTest2CellsVarying2D) {
    boundaryFactory.reset(new BoundaryFactory("periodic", deviceConfiguration));

    auto volume = volumeFactory->createConservedVolume(nx, ny, 1, ghostCells);
    volume->makeZero();

    auto rho = volume->getScalarMemoryArea("rho")->getView();
    auto mx = volume->getScalarMemoryArea("mx")->getView();
    auto my = volume->getScalarMemoryArea("my")->getView();
    auto mz = volume->getScalarMemoryArea("mz")->getView();
    auto E = volume->getScalarMemoryArea("E")->getView();

    size_t z = 0;

    for (size_t x = ghostCells; x < nx + ghostCells; ++x) {
        for (size_t y = ghostCells; y < ny + ghostCells; ++y) {

            rho.at(x, y, z) = rho.index(x, y, z);
            mx.at(x, y, z) = rho.index(x, y, z);
            my.at(x, y, z) = rho.index(x, y, z);
            mz.at(x, y, z) = rho.index(x, y, z);
            E.at(x, y, z) = rho.index(x, y, z);

        }
    }


    auto boundary = boundaryFactory->createBoundary(ghostCells);

    auto smallerGrid = alsfvm::make_shared<grid::Grid>(rvec3(0, 0, 0), rvec3(1, 1,
                0), ivec3(nx, ny, 1));
    boundary->applyBoundaryConditions(*volume, *smallerGrid);

    // X side
    for (size_t y = ghostCells; y < ny + ghostCells; ++y) {
        ASSERT_EQ(rho.at(0, y, z), rho.index(nx - 2 + ghostCells, y, z));
        ASSERT_EQ(rho.at(1, y, z), rho.index(nx - 1 + ghostCells, y, z));

        ASSERT_EQ(rho.at(ghostCells + nx + 1, y, z), rho.index(3, y, z));
        ASSERT_EQ(rho.at(ghostCells + nx, y, z), rho.index(2, y, z));
    }
}
