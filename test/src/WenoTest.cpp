#include <gtest/gtest.h>

#include "alsfvm/reconstruction/WENOCPU.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/reconstruction/WENOCoefficients.hpp"

using namespace alsfvm;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm::reconstruction;
using namespace alsfvm::grid;

TEST(WenoTest, ConstantZeroTestSecondOrder) {
    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = std::make_shared<DeviceConfiguration>();
    auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler", memoryFactory);
    WENOCPU<2> wenoCPU;

    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());

    conserved->makeZero();

    

    wenoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    for_each_internal_volume_index(*left, 0, [&](size_t , size_t middle, size_t ) {
        ASSERT_EQ(0, left->getScalarMemoryArea(0)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(1)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(2)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(3)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(4)->getPointer()[middle]);

        ASSERT_EQ(0, right->getScalarMemoryArea(0)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(1)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(2)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(3)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(4)->getPointer()[middle]);
    });
}

TEST(WenoTest, ConstantZeroTestThirdOrder) {
    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = std::make_shared<DeviceConfiguration>();
    auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler", memoryFactory);
    WENOCPU<3> wenoCPU;

    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());

    conserved->makeZero();



    wenoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    for_each_internal_volume_index(*left, 0, [&](size_t , size_t middle, size_t ) {
        ASSERT_EQ(0, left->getScalarMemoryArea(0)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(1)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(2)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(3)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(4)->getPointer()[middle]);

        ASSERT_EQ(0, right->getScalarMemoryArea(0)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(1)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(2)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(3)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(4)->getPointer()[middle]);
    });
}

TEST(WenoTest, ConstantOneTestSecondOrder) {

    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = std::make_shared<DeviceConfiguration>();
    auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler", memoryFactory);
    WENOCPU<2> wenoCPU;
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());

    for_each_cell_index(*conserved, [&] (size_t index) {
        conserved->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });
  

    wenoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    for_each_internal_volume_index(*left, 0, [&](size_t , size_t middle, size_t ) {
        ASSERT_NEAR(1, left->getScalarMemoryArea(0)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(1, left->getScalarMemoryArea(1)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(1, left->getScalarMemoryArea(2)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(1, left->getScalarMemoryArea(3)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(10, left->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);

        ASSERT_NEAR(1, right->getScalarMemoryArea(0)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(1, right->getScalarMemoryArea(1)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(1, right->getScalarMemoryArea(2)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(1, right->getScalarMemoryArea(3)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(10, right->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);
    });
}

TEST(WenoTest, ConstantOneTestThirdOrder) {

    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = std::make_shared<DeviceConfiguration>();
    auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler", memoryFactory);
    WENOCPU<3> wenoCPU;
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());

    for_each_cell_index(*conserved, [&] (size_t index) {
        conserved->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });


    wenoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    for_each_internal_volume_index(*left, 0, [&](size_t , size_t middle, size_t ) {
        ASSERT_NEAR(1, left->getScalarMemoryArea(0)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(1, left->getScalarMemoryArea(1)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(1, left->getScalarMemoryArea(2)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(1, left->getScalarMemoryArea(3)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(10, left->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);

        ASSERT_NEAR(1, right->getScalarMemoryArea(0)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(1, right->getScalarMemoryArea(1)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(1, right->getScalarMemoryArea(2)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(1, right->getScalarMemoryArea(3)->getPointer()[middle] , 1e-8);
        ASSERT_NEAR(10, right->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);
    });
}

TEST(WenoTest, ReconstructionSimple) {
    const size_t nx = 10, ny = 1, nz = 1;

    auto deviceConfiguration = std::make_shared<DeviceConfiguration>();
    auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler", memoryFactory);
    WENOCPU<2> wenoCPU;
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());
    for_each_cell_index(*conserved, [&] (size_t index) {
        // fill some dummy data
        conserved->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });

    // This is the main ingredient:
    conserved->getScalarMemoryArea("rho")->getPointer()[1] = 2;
    conserved->getScalarMemoryArea("rho")->getPointer()[2] = 0;
    conserved->getScalarMemoryArea("rho")->getPointer()[3] = 1;

    wenoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    const real epsilon = WENOCoefficients<2>::epsilon;
    const real right1 = 0.5;
    const real right2 = -1;

    const real left1 = -0.5;
    const real left2 = 1;

    const real d0 = 2.0/3.0;
    const real d1 = 1.0/3.0;

    const real beta0 = 4.0;
    const real beta1 = 1.0;

    const real alpha0 = d0 / pow(beta0 + epsilon, 2);
    const real alpha1 = d1 / pow(beta1 + epsilon, 2);
    const real alphaSum = alpha0 + alpha1;

    const real alpha0Tilde = d1 / pow(beta0 + epsilon, 2);
    const real alpha1Tilde = d0 / pow(beta1 + epsilon, 2);
    const real alphaTildeSum = alpha0Tilde + alpha1Tilde;


    const real omega0 = alpha0 / alphaSum;
    const real omega1 = alpha1 / alphaSum;

    const real omega0Tilde = alpha0Tilde / alphaTildeSum;
    const real omega1Tilde = alpha1Tilde / alphaTildeSum;

    ASSERT_EQ(omega0 * right1 + omega1 * right2, right->getScalarMemoryArea("rho")->getPointer()[2]);
    ASSERT_EQ(omega0Tilde * left1 + omega1Tilde * left2, left->getScalarMemoryArea("rho")->getPointer()[2]);

}


TEST(WenoTest, ReconstructionSimpleYDirection) {
    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = std::make_shared<DeviceConfiguration>();
    auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler", memoryFactory);
    WENOCPU<2> wenoCPU;
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz, wenoCPU.getNumberOfGhostCells());
    for_each_cell_index(*conserved, [&] (size_t index) {
        // fill some dummy data
        conserved->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });

    const size_t totalNx = nx + 2 * wenoCPU.getNumberOfGhostCells();
    // This is the main ingredient:
    conserved->getScalarMemoryArea("rho")->getPointer()[3+1 * totalNx] = 2;
    conserved->getScalarMemoryArea("rho")->getPointer()[3+2 * totalNx] = 0;
    conserved->getScalarMemoryArea("rho")->getPointer()[3+3 * totalNx] = 1;

    wenoCPU.performReconstruction(*conserved, 1, 0, *left, *right);

    const real epsilon = WENOCoefficients<2>::epsilon;
    const real right1 = 0.5;
    const real right2 = -1;

    const real left1 = -0.5;
    const real left2 = 1;

    const real d0 = 2.0/3.0;
    const real d1 = 1.0/3.0;

    const real beta0 = 4.0;
    const real beta1 = 1.0;

    const real alpha0 = d0 / pow(beta0 + epsilon, 2);
    const real alpha1 = d1 / pow(beta1 + epsilon, 2);
    const real alphaSum = alpha0 + alpha1;

    const real alpha0Tilde = d1 / pow(beta0 + epsilon, 2);
    const real alpha1Tilde = d0 / pow(beta1 + epsilon, 2);
    const real alphaTildeSum = alpha0Tilde + alpha1Tilde;


    const real omega0 = alpha0 / alphaSum;
    const real omega1 = alpha1 / alphaSum;

    const real omega0Tilde = alpha0Tilde / alphaTildeSum;
    const real omega1Tilde = alpha1Tilde / alphaTildeSum;

    ASSERT_EQ(omega0 * right1 + omega1 * right2, right->getScalarMemoryArea("rho")->getPointer()[3+2*totalNx]);
    ASSERT_EQ(omega0Tilde * left1 + omega1Tilde * left2, left->getScalarMemoryArea("rho")->getPointer()[3+2*totalNx]);

}
