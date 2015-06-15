project(gen_matrix C)

include_directories(/usr/local/opt/openblas/include
                    /usr/local/opt/lapack/include)
link_directories(/usr/local/opt/openblas/lib
                 /usr/local/opt/lapack/lib)
link_libraries(openblas lapack)

add_executable(gen gen_matrix.c)

add_executable(f2b floarid2bin.c mmio.c)