cmake_minimum_required(VERSION 3.1)
project(MEXP)

add_subdirectory(gen_matrix)
add_subdirectory(serial_mexp)
add_subdirectory(openmp)
add_subdirectory(mpi)