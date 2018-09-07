#pragma once
#include "alsuq/mpi/ExchangeCache.hpp"
#include "alsutils/mpi/Configuration.hpp"
#include "alsfvm/grid/Grid.hpp"

namespace alsuq {
namespace mpi {

class ExchangeCacheFactory {
public:
    ExchangeCacheFactory(alsutils::mpi::ConfigurationPtr configuration,
        int multiX, int multiY,
        int multiZ);


    ExchangeCachePtr makeExchangeCache(const alsfvm::grid::Grid& grid);

private:
    alsutils::mpi::ConfigurationPtr configuration;
    int multiX;
    int multiY;
    int multiZ;

    ExchangeCachePtr exchangeCache = nullptr;
};

typedef std::shared_ptr<ExchangeCacheFactory> ExchangeCacheFactoryPtr;
} // namespace mpi
} // namespace alsuq
