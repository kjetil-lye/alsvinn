#include "alsuq/run/Runner.hpp"

namespace alsuq { namespace run {

Runner::Runner(std::shared_ptr<SimulatorCreator> simulatorCreator,
               std::shared_ptr<samples::SampleGenerator> sampleGenerator,
               std::vector<size_t> sampleNumbers)
    : simulatorCreator(simulatorCreator),
      sampleGenerator(sampleGenerator),
      parameterNames(sampleGenerator->getParameterList()),
      sampleNumbers(sampleNumbers)

{

}

void Runner::run()
{
    for(size_t sample : sampleNumbers) {
        alsfvm::init::Parameters parameters;

        for (auto parameterName : parameterNames) {
            auto samples = sampleGenerator->generate(parameterName, sample);
            parameters.addParameter(name,
                                    samples);

        }

        auto simulator = simulatorCreator->createSimulator(parameters, sample);

        simulator->callWriters();
        while (!simulator->atEnd()) {
            simulator->performStep();
        }
    }
}

}
}