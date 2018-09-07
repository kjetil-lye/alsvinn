#include "alsuq/mpi/ExchangeCacheFactory.hpp"
#include "alsfvm/mpi/domain/CartesianDecomposition.hpp"
namespace alsuq {
namespace mpi {

ExchangeCachePtr ExchangeCacheFactory::makeExchangeCache(
    const alsfvm::grid::Grid& grid) {

    if (!exchangeCache) {

        std::vector<alsfvm::mpi::CellExchangerPtr> exchangers;
        alsfvm::mpi::domain::CartesianDecomposition decomposer(multiX, multiY, multiZ);

        constexpr auto numberOfSides =
            alsfvm::mpi::domain::CartesianDecomposition::numberOfSides;
        constexpr auto numberOfCorners =
            alsfvm::mpi::domain::CartesianDecomposition::numberOfCorners;


        for (size_t d = 0; d < grid.getActiveDimension(); ++d) {
            for (size_t k = 0; k < 2; ++k) {
                std::array<bool, numberOfSides> sidesToExchange;
                sidesToExchange.fill(false);
                sidesToExchange[d * 2 + k] = true;

                std::array<bool, numberOfCorners> cornersToExchange;
                cornersToExchange.fill(false);

                auto decomposeInformation = decomposer.decompose(configuration, grid,
                        sidesToExchange,
                        cornersToExchange);

                exchangers.push_back(decomposeInformation->getCellExchanger());
            }
        }

        for (size_t corner = 0; corner < numberOfCorners; ++corner) {
            std::array<bool, numberOfSides> sidesToExchange;
            sidesToExchange.fill(false);


            std::array<bool, numberOfCorners> cornersToExchange;
            cornersToExchange.fill(false);

            cornersToExchange[corner] = true;

            auto decomposeInformation = decomposer.decompose(configuration, grid,
                    sidesToExchange,
                    cornersToExchange);

            exchangers.push_back(decomposeInformation->getCellExchanger());
        }

        exchangeCache = std::make_shared<ExchangeCache>(exchangers);
    }

    return exchangeCache;
}

ExchangeCacheFactory::ExchangeCacheFactory(alsutils::mpi::ConfigurationPtr
    configuration, int multiX, int multiY, int multiZ)
    : configuration(configuration), multiX(multiX), multiY(multiY), multiZ(multiZ) {

}

}
}
