#include "alsfvm/numflux/NumericalFluxFactory.hpp"
#include "alsfvm/numflux/euler/NumericalFluxCPU.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"

#include "alsfvm/error/Exception.hpp"
#include <iostream>

namespace alsfvm { namespace numflux { 



///
/// \param equation the name of the equation (eg. Euler)
/// \param fluxname the name of the flux (eg. HLL)
/// \param reconstruction the reconstruction to use ("none" is default).
/// \param deviceConfiguration the relevant device configuration
/// \note The platform name is deduced by deviceConfiguration
///
NumericalFluxFactory::NumericalFluxFactory(const std::string& equation,
              const std::string& fluxname,
              const std::string& reconstruction,
              std::shared_ptr<DeviceConfiguration>& deviceConfiguration)
    : equation(equation), fluxname(fluxname), reconstruction(reconstruction),
      deviceConfiguration(deviceConfiguration)
{

}

///
/// Creates the numerical flux
///
NumericalFluxFactory::NumericalFluxPtr
    NumericalFluxFactory::createNumericalFlux(const grid::Grid& grid) {

    // First we must do a lot of error checking
    auto& platform = deviceConfiguration->getPlatform();
    if (platform == "cpu") {
        if (equation == "euler") {
            if (fluxname == "HLL") {
                if (reconstruction == "none") {
                    if (grid.getActiveDimension() == 3) {
                        return NumericalFluxPtr(new euler::NumericalFluxCPU<euler::HLL, 3>(grid, deviceConfiguration));
                    } else if(grid.getActiveDimension() == 2) {
                        return NumericalFluxPtr(new euler::NumericalFluxCPU<euler::HLL, 2>(grid, deviceConfiguration));
                    } else if(grid.getActiveDimension() == 1) {
                        return NumericalFluxPtr(new euler::NumericalFluxCPU<euler::HLL, 2>(grid, deviceConfiguration));
                    } else {
                        THROW("Unsupported dimension " << grid.getActiveDimension()
                              << " for equation " << equation << " platform " << platform << " and fluxname " << fluxname );
                    }
                } else {
                    THROW("Unknwon reconstruction " << reconstruction);
                }
            } else {
                THROW("Unknown flux " << fluxname);
            }
        } else {
            THROW("Unknown equation " << equation);
        }
    } else {
        THROW("Unknown platform " << platform);
    }

    THROW("Something went wrong in NumericalFluxFactory::createNumericalFlux");
}
}
}