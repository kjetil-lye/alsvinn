#include "alsfvm/boundary/BoundaryFactory.hpp"
#include "alsfvm/error/Exception.hpp"
#include "alsfvm/boundary/NeumannCPU.hpp"
namespace alsfvm { namespace boundary { 
	///
	/// Instantiates the boundary factory
	/// \param name the name of the boundary type
	/// Parameter | Description
	/// ----------|------------
	/// "neumann"   | Neumann boundary conditions
	/// "periodic"  | Periodic boundary conditions
	/// 
	/// \param deviceConfiguration the device configuration
	///
	BoundaryFactory::BoundaryFactory(const std::string& name,
		std::shared_ptr<DeviceConfiguration>& deviceConfiguration)
		: name(name), deviceConfiguration(deviceConfiguration)
	{

	}

	/// 
	/// Creates the new boundary
	/// \param ghostCellSize the number of ghost cell to use on each side.
	/// 
	std::shared_ptr<Boundary> BoundaryFactory::createBoundary(size_t ghostCellSize) {
		if (deviceConfiguration->getPlatform() == "cpu") {
			if (name == "neumann") {
				return std::shared_ptr<Boundary>(new NeumannCPU(ghostCellSize));
			}
			else {
				THROW("Unknown boundary type " << name);
			}
		}
		else {
			THROW("Unknown platform " << deviceConfiguration->getPlatform());
		}
	}
}
}