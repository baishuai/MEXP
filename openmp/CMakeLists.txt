project(openmp C)

if(APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set(CMAKE_C_COMPILER "/usr/local/bin/clang-omp")
endif()

include_directories(/usr/local/opt/openblas/include)
link_directories(/usr/local/opt/openblas/lib)
link_libraries(openblas)

set(SOURCE_FILES openmp_mexp.c)
add_executable(openmp ${SOURCE_FILES})
