#include "global.h"
#include "H5Cpp.h"
#include <iomanip>
#include <map>
#include <unistd.h>

using namespace H5;

/**
 * @brief 读取输入参数
 */
void readInputParameters()
{
    // 打开文件
    std::ifstream params("input.dat");
    if (!params.is_open())
    {
        std::cerr << "Error: 打开输入文件失败" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    std::map<std::string, std::string> input_params;
    // 读取参数
    std::string line;
    while (std::getline(params, line))
    {
        if (line.empty())
            continue;
        // 没有井号的行都视为有效输入
        if (line.find('#') == std::string::npos)
        {
            std::size_t pos = line.find("=");
            if (pos == std::string::npos)
                continue;

            std::string param = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            // 去除空格
            param.erase(0, param.find_first_not_of(" \t"));
            param.erase(param.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            input_params[param] = value;
        }
    }
    params.close();

    // 参数赋值
    U1 = std::stod(input_params["U1"]);
    V1 = std::stod(input_params["V1"]);
    U2 = std::stod(input_params["U2"]);
    V2 = std::stod(input_params["V2"]);
    W = std::stod(input_params["W"]);
    Nsp1 = std::stoi(input_params["Nsp1"]);
    Nsp2 = std::stoi(input_params["Nsp2"]);
    Nsplit = std::stoi(input_params["Nsplit"]);
    ifSaveMatrix = (input_params["ifSaveMatrix"] == "true");
    ifDiagMatrix = (input_params["ifDiagMatrix"] == "true");
}

/**
 * @brief 读入子空间基矢
 */
std::vector<long long> loadSubspace()
{
    try
    {
        H5File file("subspaces.h5", H5F_ACC_RDONLY);
        DataSet dataset = file.openDataSet(std::to_string(P));
        DataSpace dataspace = dataset.getSpace();

        int ndims = dataspace.getSimpleExtentNdims();
        hsize_t dims_out[ndims];
        dataspace.getSimpleExtentDims(dims_out, NULL);

        std::vector<long long> data(dims_out[0]);
        dataset.read(data.data(), PredType::NATIVE_LLONG);

        dataset.close();
        file.close();
        return data;
    }
    catch (const H5::Exception &e)
    {
        std::cerr << "HDF5 error loading subspace for P=" << P << ": " << e.getDetailMsg() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }
}

/**
 * @brief 保存矩阵
 */
void saveMatrix(const std::vector<int> &rows,
                const std::vector<int> &cols,
                const std::vector<complexd> &values,
                const std::string &filename)
{
    try
    {
        H5File file(filename, H5F_ACC_TRUNC);

        hsize_t dims = values.size();
        DataSpace dataspace(1, &dims);
        // 启用压缩
        DSetCreatPropList props;
        props.setDeflate(6); // 压缩级别 1-9
        // hsize_t chunk_dims[1] = {10000}; // 分块大小
        hsize_t chunk_dims[1] = {std::min((hsize_t)values.size(), (hsize_t)10000)}; // 动态分块大小
        props.setChunk(1, chunk_dims);

        DataSet ds_rows = file.createDataSet("rows", PredType::NATIVE_INT, dataspace, props);
        ds_rows.write(rows.data(), PredType::NATIVE_INT);
        DataSet ds_cols = file.createDataSet("cols", PredType::NATIVE_INT, dataspace, props);
        ds_cols.write(cols.data(), PredType::NATIVE_INT);
        // 自定义复数类型
        CompType mytype(sizeof(complexd));
        mytype.insertMember("real", 0, PredType::NATIVE_DOUBLE);
        mytype.insertMember("imag", sizeof(double), PredType::NATIVE_DOUBLE);
        DataSet ds_data = file.createDataSet("values", mytype, dataspace, props);
        ds_data.write(values.data(), mytype);

        file.close();
    }
    catch (const H5::Exception &e)
    {
        std::cerr << "HDF5 error saving matrix: " << e.getDetailMsg() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

/**
 * @brief 保存本征值和本征矢
 */
void saveEigen(const std::vector<double> &eigenvalues,
               const std::vector<std::vector<complexd>> &eigenvectors,
               const std::string &filename)
{
    try
    {
        H5File file(filename, H5F_ACC_TRUNC);

        hsize_t dims = eigenvalues.size();
        DataSpace dataspace(1, &dims);
        DataSet ds_eigenvalues = file.createDataSet("eigenvalues", PredType::NATIVE_DOUBLE, dataspace);
        ds_eigenvalues.write(eigenvalues.data(), PredType::NATIVE_DOUBLE);

        // 将本征矢量对应保存到不同的数据集
        for (size_t i = 0; i < eigenvectors.size(); i++)
        {
            std::string dataset_name = "eigenvector_" + std::to_string(i);
            hsize_t vec_dims = eigenvectors[i].size();
            DataSpace vec_dataspace(1, &vec_dims);
            // 自定义复数类型
            CompType mytype(sizeof(complexd));
            mytype.insertMember("real", 0, PredType::NATIVE_DOUBLE);
            mytype.insertMember("imag", sizeof(double), PredType::NATIVE_DOUBLE);
            DataSet ds_eigenvector = file.createDataSet(dataset_name, mytype, vec_dataspace);
            ds_eigenvector.write(eigenvectors[i].data(), mytype);
        }

        file.close();
    }
    catch (const H5::Exception &e)
    {
        std::cerr << "HDF5 error saving matrix: " << e.getDetailMsg() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

/**
 * @brief 追加保存基态能量和基态矢量
 */
void appendGroundState(double groundEnergy, const std::vector<complexd> &groundVector, int P)
{
    // 追加基态能量到txt文件
    std::ofstream engFile("./Eigen/groundeng.txt", std::ios::app);
    if (!engFile.is_open())
    {
        std::cerr << "Error: 无法打开基态能量文件" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    engFile << "P=" << P << ", E=" << std::setprecision(16) << groundEnergy << std::endl;
    engFile.close();

    // 追加基态矢量到HDF5文件
    try
    {
        H5File file;
        // 检查文件是否存在
        if (access("./Eigen/eigenvectors.h5", F_OK) == 0)
        {
            // 文件存在，以读写模式打开
            file.openFile("./Eigen/eigenvectors.h5", H5F_ACC_RDWR);
        }
        else
        {
            // 文件不存在，创建新文件
            file = H5File("./Eigen/eigenvectors.h5", H5F_ACC_TRUNC);
        }

        // 创建数据集名称
        std::string dataset_name = "subspace_" + std::to_string(P);
        hsize_t vec_dims = groundVector.size();
        DataSpace vec_dataspace(1, &vec_dims);

        // 自定义复数类型
        CompType mytype(sizeof(complexd));
        mytype.insertMember("real", 0, PredType::NATIVE_DOUBLE);
        mytype.insertMember("imag", sizeof(double), PredType::NATIVE_DOUBLE);

        DataSet ds_eigenvector = file.createDataSet(dataset_name, mytype, vec_dataspace);
        ds_eigenvector.write(groundVector.data(), mytype);

        file.close();
    }
    catch (const H5::Exception &e)
    {
        std::cerr << "HDF5 error appending ground state: " << e.getDetailMsg() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
