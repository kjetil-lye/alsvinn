cmake_minimum_required (VERSION 3.1.0)

FILE(GLOB_RECURSE SRC src/*.cpp)

FILE(GLOB_RECURSE EXAMPLES examples/*.py examples/*.xml)

ADD_EXECUTABLE(alsuqcli ${SRC} ${EXAMPLES})

if(ALSVINN_MPI_GPU_DIRECT)
        message(STATUS "Using gpu direct alsuqcli")
        # mpich is really strict on this, gpu direct will be disabled
        # o therwise.
        # yes, it's an ugly hack, but it's a working hack.
  SET(CUDARTLIB -lcudart)
ELSE()
        SET(CUDARTLIB "")
endif()

TARGET_LINK_LIBRARIES(alsuqcli alsuq Boost::program_options ${OpenMP_CXX_LIB_NAMES} ${CUDARTLIB})

INSTALL(TARGETS
    alsuqcli
    EXPORT alsvinn-targets
    RUNTIME
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT alsuqcli)
