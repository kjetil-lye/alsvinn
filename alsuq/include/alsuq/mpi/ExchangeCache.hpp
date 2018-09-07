#pragma once
#include "alsfvm/volume/VolumePair.hpp"
#include "alsfvm/mpi/Request.hpp"
#include "alsfvm/mpi/CellExchanger.hpp"
namespace alsuq {
namespace mpi {

//! Does the border exchange for the statistics classes. It caches the results, so
//! that it can be reused by several statistics classes.
class ExchangeCache {
public:
    ExchangeCache(std::vector<alsfvm::mpi::CellExchangerPtr>& exchangers);
    void startExchange(const alsfvm::volume::Volume& conservedInput,
        const alsfvm::volume::Volume& extraInput,
        size_t ghostSize,
        size_t side,
        int timestep);


    alsfvm::volume::VolumePair getBorder(size_t side, int timestep);


    bool hasSide(size_t side) const;
private:
    std::array<int, 6> currentTimestepPerSide{-1, -1, -1, -1, -1, -1};

    std::array<std::array<alsfvm::mpi::RequestContainer, 2>, 6> requests;
    std::vector<alsfvm::volume::VolumePair> volumeCache;
    std::vector<alsfvm::mpi::CellExchangerPtr> exchangers;

    void initializeVolumeCache(const alsfvm::volume::Volume& input,
        size_t ghostSize, size_t side);

};

typedef std::shared_ptr<ExchangeCache> ExchangeCachePtr;
} // namespace mpi
} // namespace alsuq
