# MEXP using OpenMP and MPI


基于矩阵幂的对称正定线性方程组求解的并行算法分析研究

OpenMP实现与MPI+OpenMP实现


### 编译环境

性能测试使用Intel编译器icc，并且依赖Intel MKL数学库,测试前请确保已经正确安装了Intel MKL库

编译代码需要执行MKL库中相关的环境变量设置

具体在“探索100”中需要的设置为

```sh
#intel openmp
source /apps/intel/Compiler/11.1/069/c/bin/iccvars.sh intel64
source /apps/intel/Compiler/11.1/069/f/bin/ifortvars.sh intel64

#intel mkl
source /apps/intel/Compiler/11.1/069/c/mkl/tools/environment/mklvarsem64t.sh

#intel openmpi
source /apps/intel/Compiler/11.1/069/c/bin/iccvars.sh intel64
source /apps/intel/Compiler/11.1/069/f/bin/ifortvars.sh intel64
export LD_LIBRARY_PATH=/apps/mpi/openmpi-1.4.3-intel11.1/lib:$LD_LIBRARY_PATH
export PATH=/apps/mpi/openmpi-1.4.3-intel11.1/bin:$PATH
export LD_LIBRARY_PATH=/apps/intel/Compiler/11.1/069/f/lib/intel64:$LD_LIBRARY_PATH

```

### 编译命令

编译OpenMP版本的并行代码命令如下
```sh
icc -o prog-omp -openmp -mkl prog-omp.c
```

编译MPI+OpenMP的程序命令如下
```sh
mpicc -o prog-mpi -mkl -openmp -o prog-mpi.c
```

下面给出各个程序的具体编译命令

```sh
#floarid2bin
icc -mkl -o f2b floarid2bin.c mmio.c

#gen_matrix
icc -mkl -o gen gen_matrix.c



#openmp
icc -mkl -openmp -o openmp_all openmp_all.c



#mpi_cg
mpicc -mkl -openmp -o mpi_cg mpi_cg.c

#mpi_mexp
mpicc -mkl -openmp -o mpi_mexp mpi_mexp.c

#mpi_all
mpicc -mkl -openmp -o mpi_all mpi_all.c

```

### 测试数据

data目录中给出了部分从佛罗里达矩阵集中下载的数据，解压缩后可以通过 floarid2bin 程序将其转化为实验所需的数据格式

### 各个程序说明以及运行时参数

#### gen_matrix

用来生成测试使用的密集矩阵，其输出符合测试程序对矩阵格式的要求

运行参数 Usage: gen [-m] -n num [-t num] -f string

`-n`：正整数，指定生成矩阵的规模

`-f`：字符串，用于给生成矩阵命名添加后缀

`-m`：有两种生成矩阵的方法，无`-m`参数时使用方法一，有`-m`时使用方法二

`-t`：正整数，用于指定方法二中的参数alpha，alpha取值此值的以10为底的幂值


#### f2b

转化佛罗里达矩阵为实验使用格式，输出名为输入名加上`.bin`后缀

参数 Usage: f2b -f filename

`-f`：字符串，指定输入矩阵名称

#### openmp_all

使用OpenMP实现的并行MEXP算法以及CG算法

参数  Usage: mexp -t num_threads -f string

`-f`：字符串，指定输入矩阵名称

`-t`：正整数，指定OpenMP使用的线程数量


#### mpi_cg 与 mpi_mexp

使用MPI+Open MP实现的并行CG算法和并行MEXP算法

Usage: mpi_cg -f string -t num_threads

`-f`：字符串，指定输入矩阵名称

`-t`：正整数，指定OpenMP使用的线程数量

MPI进程数量使用 mpirun -np指定，或者通过作业管理系统LSF等指定。


#### mpi_all

这个是为了便于在测试时批量执行测试样例，可以修改源程序中`files`变量测试多组数据，数据需要放在同一文件夹中。

`-f`：字符串，指定输入矩阵文件夹名，必须以`/`结尾

`-t`：正整数，指定OpenMP使用的线程数量
