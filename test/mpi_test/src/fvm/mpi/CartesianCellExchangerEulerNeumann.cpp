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

//! This file tests almost the same as CartesianCellExchanger, only with the Euler
//! variables (there was a bug earlier where the simulation would hang, therefore this is tested,
//! also it helps getting the exchange tested for more than one pair of variables)
//!
//! This tests with neumann boundaries, the other with periodic
#include <gtest/gtest.h>
#include "alsfvm/mpi/domain/CartesianDecomposition.hpp"
#include "alsfvm/volume/make_volume.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "do_serial.hpp"
#include "alsfvm/mpi/CartesianCellExchanger.hpp"

using namespace alsfvm;

TEST(CartesianCellExchangerEulerNeumann, Test1D) {
    auto mpiConfiguration = alsfvm::make_shared<alsfvm::mpi::Configuration>
        (MPI_COMM_WORLD);
    const int numberOfProcessors = mpiConfiguration->getNumberOfProcesses();
    const int rank = mpiConfiguration->getRank();


    const int N = 16 * numberOfProcessors;

    rvec3 lowerCorner = {0, 0, 0};
    rvec3 upperCorner = {1, 0, 0};

    const std::string platform = "cpu";
    const std::string equation = "euler1";

    const int ghostCells = 3;


    auto grid = alsfvm::make_shared<grid::Grid>(lowerCorner, upperCorner, ivec3{N * numberOfProcessors, 1, 1},
            boundary::allNeumann());


    auto volume = volume::makeConservedVolume(platform,
            equation,
    {N, 1, 1},
    ghostCells);


    alsfvm::mpi::domain::CartesianDecomposition decomposer(numberOfProcessors, 1,
        1);

    auto information = decomposer.decompose(mpiConfiguration, *grid);

    auto newGrid = information->getGrid();
    auto newDimensions = newGrid->getDimensions();
    ASSERT_EQ(N, newDimensions.x);
    ASSERT_EQ(1, newDimensions.y);
    ASSERT_EQ(1, newDimensions.z);


    for (int side = 0; side < 6; ++side) {
        if (side == 0 && rank == 0) {
            ASSERT_EQ(boundary::NEUMANN, newGrid->getBoundaryCondition(side))
                    << "side = " << side << "\n"
                        << "rank = " << rank << "\n";;
        } else if (side == 1 && rank == numberOfProcessors - 1) {
            ASSERT_EQ(boundary::NEUMANN, newGrid->getBoundaryCondition(side))
                    << "side = " << side << "\n"
                        << "rank = " << rank << "\n";;
        } else {
            ASSERT_EQ(boundary::MPI_BC, newGrid->getBoundaryCondition(side))
                    << "side = " << side << "\n"
                        << "rank = " << rank << "\n";
        }
    }



    auto valueAtIndex = [&](int i,  int rank, size_t var) {
        return (i - ghostCells + N * rank) * volume->getNumberOfVariables() + var;
    };

    for (size_t var = 0; var < volume->getNumberOfVariables(); ++var) {
        for (int i = ghostCells; i < N + ghostCells; ++i) {
            (*volume->getScalarMemoryArea(var))[i] = valueAtIndex(i, rank, var);
        }
    }

    const real magicValue = 42 * numberOfProcessors + rank;

    for (size_t var = 0; var < volume->getNumberOfVariables(); ++var) {
        for (int i = 0; i < ghostCells; ++i) {
            (*volume->getScalarMemoryArea(var))[i] = magicValue;
            (*volume->getScalarMemoryArea(var))[N + ghostCells + i] = magicValue;
        }
    }


    for (size_t var = 0; var < volume->getNumberOfVariables(); ++var) {
        for (int i = ghostCells; i < N + ghostCells; ++i) {
            auto value = (*volume->getScalarMemoryArea(var))[i];
            ASSERT_EQ(valueAtIndex(i, rank, var), value);
        }
    }

    // Make sure max works
    real waveSpeed = 42 * rank;
    real maxWaveSpeed = information->getCellExchanger()->adjustWaveSpeed(waveSpeed);

    ASSERT_EQ(42 * (numberOfProcessors - 1), maxWaveSpeed);


    information->getCellExchanger()->exchangeCells(*volume, *volume).waitForAll();


    auto rankIndex = [&](int x) {
        const int nx = numberOfProcessors;

        if (x < 0) {
            x += nx;
        }

        if (x > nx - 1) {
            x -= nx;
        }

        return x;

    };

    if (numberOfProcessors == 1) {
        return;
    }



    // left side
    for (size_t var = 0; var < volume->getNumberOfVariables(); ++var) {
        for (int i = 0; i < ghostCells; ++i) {
            int index = i;
            auto value = (*volume->getScalarMemoryArea(var))[index];

            int expectedValue = valueAtIndex(N + i, rankIndex(rank - 1), var);


            if (rank == 0) {
                expectedValue = magicValue;
            }

            EXPECT_EQ(expectedValue, value)
                    << "Failed at left ghost index " << i << "  on processor " << rank
                        << "\nvar = " << volume->getName(var) << "(" << var << ")";
        }

        // right side
        for (int i = 0; i < ghostCells; ++i) {
            int index = N + ghostCells + i;
            auto value = (*volume->getScalarMemoryArea(var))[index];
            int expectedValue = valueAtIndex(i + ghostCells, rankIndex(rank + 1), var);

            if (rank == numberOfProcessors - 1) {
                expectedValue = magicValue;
            }

            EXPECT_EQ(expectedValue, value)
                    << "Failed at right ghost index " << i << "  on processor " << rank
                        << "\nvar = " << volume->getName(var) << "(" << var << ")";
        }
    }

#if 0 //debug output
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < N + 2 * ghostCells; ++i) {
            auto value = (*volume->getScalarMemoryArea(0))[i];
            std::cout << value << std::endl;
        }

        std::cout << "_______________________________" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        for (int i = 0; i < N + 2 * ghostCells; ++i) {
            auto value = (*volume->getScalarMemoryArea(0))[i];
            std::cout << value << std::endl;
        }

        std::cout << "_______________________________" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
#endif

}

TEST(CartesianCellExchangerEulerNeumann, Test2D) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto mpiConfiguration = alsfvm::make_shared<alsfvm::mpi::Configuration>
        (MPI_COMM_WORLD);
    const int numberOfProcessors = mpiConfiguration->getNumberOfProcesses();
    const int rank = mpiConfiguration->getRank();


    const int N = 8 * numberOfProcessors;

    if (numberOfProcessors < 4) {
        return;
    }


    rvec3 lowerCorner = {0, 0, 0};
    rvec3 upperCorner = {1, 1, 0};

    const std::string platform = "cpu";
    const std::string equation = "euler2";

    const int ghostCells = 3;


    int nx = numberOfProcessors;
    int ny = 1;

    while (nx / ny > 2) {
        nx = nx / 2;
        ny = ny * 2;
    }

    ASSERT_EQ(nx * ny, numberOfProcessors);
    auto grid = alsfvm::make_shared<grid::Grid>(lowerCorner, upperCorner, ivec3{N * nx, N * ny, 1},
            boundary::allNeumann());


    auto volume = volume::makeConservedVolume(platform,
            equation,
    {N, N, 1},
    ghostCells);


    alsfvm::mpi::domain::CartesianDecomposition decomposer(nx, ny, 1);

    auto information = decomposer.decompose(mpiConfiguration, *grid);

    auto newGrid = information->getGrid();
    auto newDimensions = newGrid->getDimensions();
    ASSERT_EQ(N, newDimensions.x);
    ASSERT_EQ(N, newDimensions.y);
    ASSERT_EQ(1, newDimensions.z);




    const int M = N + 2 * ghostCells;


    // Set a default magicVAlue to make debugging easier
    const real magicValue = N * N * 42 * numberOfProcessors + rank;

    for (size_t var = 0; var < volume->getNumberOfVariables(); ++var) {
        for (int i = 0; i < M * M; ++i) {
            (*volume->getScalarMemoryArea(var))[i] = magicValue;
        }
    }

    // Computes the x component of the rank
    auto xComponent = [&](int r) {
        return r % nx;
    };

    // computes the y component of the rank
    auto yComponent = [&](int r) {
        return r / nx;
    };

    auto computeValue = [&](int i, int j, int r, size_t var) {

        return ((i - ghostCells + xComponent(r) * N) + (j - ghostCells + yComponent(
                        r) * N) * N) * volume->getNumberOfVariables() + var;
    };

    auto rankIndex = [&](int x, int y) {
        if (x < 0) {
            x += nx;
        }

        if (x > nx - 1) {
            x -= nx;
        }

        if (y < 0) {
            y += ny;
        }

        if (y > ny - 1) {
            y -= ny;
        }

        return x + y * nx;
    };

    for (size_t var = 0; var < volume->getNumberOfVariables(); ++var) {
        for (int i = ghostCells; i < N + ghostCells; ++i) {
            for (int j = ghostCells; j < N + ghostCells; ++j) {
                (*volume->getScalarMemoryArea(var))[j * M + i] = computeValue(i, j, rank, var);
            }
        }
    }



    for (size_t var = 0; var < volume->getNumberOfVariables(); ++var) {
        for (int i = ghostCells; i < N + ghostCells; ++i) {
            for (int j = ghostCells; j < N + ghostCells; ++j) {
                auto value = (*volume->getScalarMemoryArea(var))[j * M + i];
                ASSERT_EQ(computeValue(i, j, rank, var), value);
            }
        }
    }

    // Make sure max works
    real waveSpeed = 42 * rank;
    real maxWaveSpeed = information->getCellExchanger()->adjustWaveSpeed(waveSpeed);

    ASSERT_EQ(42 * (numberOfProcessors - 1), maxWaveSpeed);

    const int xRank = xComponent(rank);
    const int yRank = yComponent(rank);

    for (int side = 0; side < 6; ++side) {
        if (side == 0 && xRank == 0) {
            ASSERT_EQ(boundary::NEUMANN, newGrid->getBoundaryCondition(side));
        } else if (side == 1 && xRank == nx - 1) {
            ASSERT_EQ(boundary::NEUMANN, newGrid->getBoundaryCondition(side));
        } else if (side == 2 && yRank == 0 ) {
            ASSERT_EQ(boundary::NEUMANN, newGrid->getBoundaryCondition(side));
        } else if (side == 3 && yRank == ny - 1) {
            ASSERT_EQ(boundary::NEUMANN, newGrid->getBoundaryCondition(side));
        } else {
            ASSERT_EQ(boundary::MPI_BC, newGrid->getBoundaryCondition(side));
        }
    }

    auto neighbours =
        alsfvm::dynamic_pointer_cast<alsfvm::mpi::CartesianCellExchanger>
        (information->getCellExchanger())->getNeighbours();

    if (xRank != 0) {
        ASSERT_EQ(rankIndex(xRank - 1, yRank), neighbours[0])
                << "Failed with"
                    << "\n\trank              = " << rank
                    << "\n\tnx                = " << nx
                    << "\n\tny                = " << ny
                    << "\n\txRank             = " << xRank
                    << "\n\tyRank             = " << yRank
                    << "\n\tumberOfProcessors = " << numberOfProcessors;
    } else {
        ASSERT_EQ(-1, neighbours[0])
                << "Failed with"
                    << "\n\trank              = " << rank
                    << "\n\tnx                = " << nx
                    << "\n\tny                = " << ny
                    << "\n\txRank             = " << xRank
                    << "\n\tyRank             = " << yRank
                    << "\n\tumberOfProcessors = " << numberOfProcessors;
    }


    if (xRank != nx - 1) {
        ASSERT_EQ(rankIndex(xRank + 1, yRank), neighbours[1])
                << "Failed with"
                    << "\n\trank              = " << rank
                    << "\n\tnx                = " << nx
                    << "\n\tny                = " << ny
                    << "\n\txRank             = " << xRank
                    << "\n\tyRank             = " << yRank
                    << "\n\tumberOfProcessors = " << numberOfProcessors;
    } else {
        ASSERT_EQ(-1, neighbours[1])
                << "Failed with"
                    << "\n\trank              = " << rank
                    << "\n\tnx                = " << nx
                    << "\n\tny                = " << ny
                    << "\n\txRank             = " << xRank
                    << "\n\tyRank             = " << yRank
                    << "\n\tumberOfProcessors = " << numberOfProcessors;
    }


    if (yRank != 0) {
        ASSERT_EQ(rankIndex(xRank, yRank - 1), neighbours[2])
                << "Failed with"
                    << "\n\trank              = " << rank
                    << "\n\tnx                = " << nx
                    << "\n\tny                = " << ny
                    << "\n\txRank             = " << xRank
                    << "\n\tyRank             = " << yRank
                    << "\n\tumberOfProcessors = " << numberOfProcessors;
    } else {
        ASSERT_EQ(-1, neighbours[2])
                << "Failed with"
                    << "\n\trank              = " << rank
                    << "\n\tnx                = " << nx
                    << "\n\tny                = " << ny
                    << "\n\txRank             = " << xRank
                    << "\n\tyRank             = " << yRank
                    << "\n\tumberOfProcessors = " << numberOfProcessors;
    }


    if (yRank != ny - 1) {
        ASSERT_EQ(rankIndex(xRank, yRank + 1), neighbours[3])
                << "Failed with"
                    << "\n\trank              = " << rank
                    << "\n\tnx                = " << nx
                    << "\n\tny                = " << ny
                    << "\n\txRank             = " << xRank
                    << "\n\tyRank             = " << yRank
                    << "\n\tumberOfProcessors = " << numberOfProcessors;
    } else {
        ASSERT_EQ(-1, neighbours[3])
                << "Failed with"
                    << "\n\trank              = " << rank
                    << "\n\tnx                = " << nx
                    << "\n\tny                = " << ny
                    << "\n\txRank             = " << xRank
                    << "\n\tyRank             = " << yRank
                    << "\n\tumberOfProcessors = " << numberOfProcessors;
    }


    //ASSERT_EQ((((rank%nx)+1)%nx) + (rank/nx)*nx, neighbours[0]);

    information->getCellExchanger()->exchangeCells(*volume, *volume).waitForAll();


    if (numberOfProcessors == 1) {
        return;
    }

#if 0 //debug output
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < M * M; ++i) {
            auto value = (*volume->getScalarMemoryArea(0))[i];
            std::cout << value << std::endl;
        }

        std::cout << "_______________________________" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        for (int i = 0; i < M * M; ++i) {
            auto value = (*volume->getScalarMemoryArea(0))[i];
            std::cout << value << std::endl;
        }

        std::cout << "_______________________________" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
#endif

    for (size_t var = 0; var < volume->getNumberOfVariables(); ++var) {

        // left side
        for (int i = 0; i < ghostCells; ++i) {
            for (int j = ghostCells; j < N + ghostCells; ++j) {
                int index = i + j * M;
                auto value = (*volume->getScalarMemoryArea(var))[index];

                int expectedValue = computeValue((i + N), j, rankIndex(xRank - 1, yRank), var);


                if (xRank == 0 ) {
                    expectedValue = magicValue;
                }

                ASSERT_EQ(expectedValue, value)
                        << "Failed at left ghost index " << i << " and j = " << j << "  on processor "
                            << rank;
            }
        }

        // right side
        for (int i = N + ghostCells; i < M; ++i) {
            for (int j = ghostCells; j < N + ghostCells; ++j) {
                int index = i + j * M;
                auto value = (*volume->getScalarMemoryArea(var))[index];

                int expectedValue = computeValue((i - N), j, rankIndex(xRank + 1, yRank), var);

                if (xRank == nx - 1 ) {
                    expectedValue = magicValue;
                }

                ASSERT_EQ(expectedValue, value)
                        << "Failed at right ghost index " << i << " and j = " << j << "  on processor "
                            << rank;
            }
        }


        // bottom side
        for (int i = ghostCells; i < N + ghostCells; ++i) {
            for (int j = 0; j < ghostCells; ++j) {
                int index = i + j * M;
                auto value = (*volume->getScalarMemoryArea(var))[index];

                int expectedValue = computeValue(i, j + N, rankIndex(xRank, yRank - 1), var);

                if (yRank == 0 ) {
                    expectedValue = magicValue;
                }

                ASSERT_EQ(expectedValue, value)
                        << "Failed at bottom ghost index " << i << " and j = " << j << "  on processor "
                            << rank;
            }
        }

        // top side
        for (int i = ghostCells; i < N + ghostCells; ++i) {
            for (int j = N + ghostCells; j < M; ++j) {
                int index = i + j * M;
                auto value = (*volume->getScalarMemoryArea(var))[index];

                int expectedValue = computeValue(i, j - N, rankIndex(xRank, yRank + 1), var);

                if (yRank == ny - 1 ) {
                    expectedValue = magicValue;
                }

                ASSERT_EQ(expectedValue, value)
                        << "Failed at top ghost index " << i << " and j = " << j << "  on processor " <<
                            rank;
            }
        }
    }
}
