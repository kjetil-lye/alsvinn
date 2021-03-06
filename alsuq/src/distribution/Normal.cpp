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

#include "alsuq/distribution/Normal.hpp"
#include "alsutils/error/Exception.hpp"

namespace alsuq {
namespace distribution {
Normal::Normal(const Parameters& parameters)
    : mean(parameters.getDouble("mean")),
      standardDeviation(parameters.getDouble("sd")) {
    // The problem with normal distribution is that it needs to regenerate the samples
    // which does not combine well with how we generate samples at the moment (see alsuq/src/generator/STLMersenne.cpp)
    //
    // We could do the ppf (inverse of normal PDF), but it is difficult to implement in  C++,
    // therefore, we just let users do the heavy lifting in Python. (will be faster than calling the python function from C++)
    THROW("Normal variables are not supported in the current implementation."
        << " Use Uniform and scipy.stats.norm.ppf(X) where X is the random viriable.");

}

real Normal::generate(generator::Generator& generator, size_t component,
    size_t sample) {
    THROW("Normal variables are not supported in the current implementation."
        << " Use Uniform and scipy.stats.norm.ppf(X) where X is the random viriable.");

    if (hasBuffer) {

        hasBuffer = false;
        return scale(buffer);
    }

    else {
        real U1, U2, r;

        // rejection method to obtain (U1,U2) - uniform random variable on unit circle
        do {
            // draw two uniform random variable on [-1, 1]
            U1 = 2 * generator.generate(component, sample) - 1;
            U2 = 2 * generator.generate(component, sample) - 1;
            // compute radius
            r = square(U1) + square(U2);

        } while (r > 1.0);

        // Box-Muller transformation
        real w  = sqrt(-2 * log(r) / r);
        real N1 = U1 * w;
        real N2 = U2 * w;

        buffer = N2;
        hasBuffer = true;

        return scale(N1);
    }
}

real Normal::scale(real x) {
    return x * standardDeviation + mean;
}
}
}
