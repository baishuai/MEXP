cmake_minimum_required(VERSION 3.1)
project(mpi)

find_package(MPI REQUIRED)

set(MPI_COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -fopenmp")

include_directories(/usr/local/opt/openblas/include)
link_directories(/usr/local/opt/openblas/lib)
link_libraries(openblas iomp5)
include_directories(${MPI_INCLUDE_PATH})

macro(ADD_MPI_EXEC TEST_NAME)
    add_executable(mpi_${TEST_NAME} ${TEST_NAME}.c)
    target_link_libraries(mpi_${TEST_NAME} ${MPI_LIBRARIES})

    if(MPI_COMPILE_FLAGS)
      set_target_properties(mpi_${TEST_NAME} PROPERTIES
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif()

    if(MPI_LINK_FLAGS)
      set_target_properties(mpi_${TEST_NAME} PROPERTIES
        LINK_FLAGS "${MPI_LINK_FLAGS}")
    endif()
endmacro()

ADD_MPI_EXEC(mpi_cg)
ADD_MPI_EXEC(mpi_mexp)
ADD_MPI_EXEC(mpi_all)