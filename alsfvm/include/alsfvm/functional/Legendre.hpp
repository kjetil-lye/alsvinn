#pragma once
#include "alsfvm/functional/Functional.hpp"
namespace alsfvm { namespace functional { 

    //! Computes the spatial integral of the n-th Legendre polynomial,
    //! ie it will compute
    //!
    //! \f[\int_D L_n(u(x))\; dx\f]
    //!
    //! where \f$L_n\f$ is the n-th Legendre polynomial.
    //!
    //! You have the option of scaling the input u to the interval [-1,1] by specifying the keywords
    //! *maxValue* and *minValue*, this will essentially compute
    //!
    //! \f[\int_D L_n(\frac{u(x)-\mathrm{minValue}}{\mathrm{maxValue}-\mathrm{minValue}})\;dx\f]
    //!
    //! \note The computation of the Legendre polynomials are done through the boost library,
    //!        see http://www.boost.org/doc/libs/1_46_1/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/legendre.html
    //!        for any implementation details. In short, \f$L_n(x)=boost::math::legendre_p(n,x)\f$
    //!
    class Legendre : public Functional {
    public:
        //! The following parameters are accepted through parameters
        //!
        //!    Name     | Description
        //!    ---------+-------------
        //!    minValue | minimum value that the solution can obtain
        //!    maxValue | maximum value that the solution can obtain
        //!    degree   | the degree of the polynomial
        //!    variables| the variables to compute for (space separated)
        Legendre(const Parameters& parameters);

        //! Computes the operator value on the givne input data
        //!
        //! @note In order to support time integration, the result should be
        //!       added to conservedVolumeOut and extraVolumeOut, not overriding
        //!       it.
        //!
        //! @param[out] conservedVolumeOut at the end, should have the contribution
        //!             of the functional for the conservedVariables
        //!
        //! @param[out] extraVolumeOut at the end, should have the contribution
        //!             of the functional for the extraVariables
        //!
        //! @param[in] conservedVolumeIn the state of the conserved variables
        //!
        //! @param[in] extraVolume the state of the extra volume
        //!
        //! @param[in] weight the current weight to be applied to the functional. Ie, the functional should compute
        //!                   \code{.cpp}
        //!                   conservedVolumeOut += weight + f(conservedVolumeIn)
        //!                   \endcode
        //!
        virtual void operator()(volume::Volume& conservedVolumeOut,
                        volume::Volume& extraVolumeOut,
                        const volume::Volume& conservedVolumeIn,
                        const volume::Volume& extraVolumeIn,
                        const real weight,
                        const grid::Grid& grid
                        ) override;

        //! Returns ivec3{1,1,1} -- we only need one element to represent this functional
        virtual ivec3 getFunctionalSize(const grid::Grid& grid) const override;


    private:
        const real minValue = -1;
        const real maxValue = 1;
        const int degree = 1;

        std::vector<std::string> variables;
    };
} // namespace functional
} // namespace alsfvm
