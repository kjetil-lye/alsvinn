#include "alsuq/mpi/ExchangeCache.hpp"

namespace alsuq {
namespace mpi {

ExchangeCache::ExchangeCache(std::vector<alsfvm::mpi::CellExchangerPtr>&
    exchangers)
    : exchangers(exchangers) {

}

void ExchangeCache::startExchange(const alsfvm::volume::Volume& conservedInput,
    const alsfvm::volume::Volume& extraInput,
    size_t ghostSize,
    size_t side, int timestep) {

    if (volumeCache.size() <= side
        || !volumeCache[side].getConservedVolume()) {
        initializeVolumeCache(conservedInput, ghostSize, side);
    }

    if (currentTimestepPerSide[side] > timestep) {
        THROW("Something wrong happened in the the ExchangeCache class."
            << "We were asked for a timestep that has already passed. "
            << "\tside   = " << side << "\n"
            << "\ttimestep = " << timestep << "\n"
            << "\tcurrentTimestepPerSide[side]" << currentTimestepPerSide[side] << "\n");
    }

    if (currentTimestepPerSide[side] < timestep) {
        requests[side][0] = exchangers[side]->exchangeCells(
                *volumeCache[side].getConservedVolume(),
                conservedInput
            );
        requests[side][1] = exchangers[side]->exchangeCells(
                *volumeCache[side].getExtraVolume(),
                extraInput
            );

    }

}

alsfvm::volume::VolumePair ExchangeCache::getBorder(size_t side, int timestep) {

    if (currentTimestepPerSide[side] != timestep) {
        THROW("Something wrong happened in the the ExchangeCache class."
            << "We were asked for a timestep is not being transfered "
            << "\tside   = " << side << "\n"
            << "\ttimestep = " << timestep << "\n"
            << "\tcurrentTimestepPerSide[side]" << currentTimestepPerSide[side] << "\n");
    }

    requests[side][0].waitForAll();
    requests[side][1].waitForAll();

    return volumeCache[side];

}

bool ExchangeCache::hasSide(size_t side) const {
    for (const auto& exchanger : exchangers) {
        if (exchanger->getNeighbours()[side] > 0) {
            return true;
        }
    }

    return false;
}

void ExchangeCache::initializeVolumeCache(const alsfvm::volume::Volume&
    input, size_t ghostSize, size_t side) {

    const size_t nx = input.getTotalNumberOfXCells();
    const size_t ny = input.getTotalNumberOfYCells();
    const size_t nz = input.getTotalNumberOfZCells();

    const size_t direction = side / 2;
    const size_t nxSide = direction == 0 ? ghostSize : nx;
    const size_t nySide = direction == 1 ? ghostSize : ny;
    const size_t nzSide = direction == 2 ? ghostSize : nz;
    auto conserved = input.makeInstance(nxSide, nySide,
            nzSide);
    auto extra = input.makeInstance(nxSide, nySide, nzSide);

    volumeCache[side] = alsfvm::volume::VolumePair(conserved, extra);

}
}
}
