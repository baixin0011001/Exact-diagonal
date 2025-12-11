#需要先使用conda安装hdf5

# HDF5库文件
HDF5_DIR = /public/home/baixing/miniconda3/envs/hdf5
HDF5_INC = $(HDF5_DIR)/include
HDF5_LIB = $(HDF5_DIR)/lib

# eigen库
EIGEN_DIR=/public/home/baixing/cpplib/eigen/include/eigen3

# Spectra库
SPECTRA_DIR=/public/home/baixing/cpplib/spectra/include

# 编译器及其选项
MPICXX = mpicxx
CXXFLAGS = -I$(HDF5_INC) -I$(EIGEN_DIR) -I$(SPECTRA_DIR) -Wl,-rpath,$(HDF5_LIB)
LDFLAGS = -L$(HDF5_LIB) -lhdf5 -lhdf5_cpp -lmpi -ldl -lz

# 源文件和目标文件
SOURCES = global.cpp io.cpp model.cpp loop.cpp main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = ExSolver

#加载模块的命令
LOAD_MODULE = module load mpi/intelmpi/2017.4.239

# 默认规则
all: $(TARGET)

# 链接规则
$(TARGET): $(OBJECTS)
	$(LOAD_MODULE) && $(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# 编译规则
%.o: %.cpp
	$(LOAD_MODULE) && $(MPICXX) $(CXXFLAGS) -c $< -o $@

# 清理生成文件
clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
