#include <iostream>
#include <chrono>
#include <sys/stat.h>
#include "global.h"
#include "model.h"
#include <unistd.h>

std::string get_current_time();
void checkFolder(const std::string &path);

int main(int argc, char **argv)
{
    // ============================== 初始化 ============================
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 读取输入参数
    readInputParameters();
    gamma1 = Nsp1 * 2.0 * PI / Nsplit;
    gamma2 = Nsp2 * 2.0 * PI / Nsplit;

    // 对角化单体哈密顿量
    int s = 1; // 能带指标
    diagH0(s);

    // 清空文件夹
    if (rank == 0)
    {
        checkFolder("./Hsave");
        checkFolder("./Eigen");
        std::cout << std::endl;
        std::cout << "初始化完成, 当前时间: " << get_current_time();
        std::cout << "进程数量: " << size << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ============================== 主循环：只计算和保存哈密顿量 ============================
    for (P = 0; P < Ns; P++)
    {
        // 加载子空间
        subspace = loadSubspace();
        // 划分任务
        int part_begin, part_end;
        int dim = subspace.size();
        int part = dim / size;
        if (part * size < dim && part > size)
            part++;
        part_begin = part * rank;
        if (rank == size - 1)
            part_end = dim;
        else
            part_end = part * (rank + 1);

        if (rank == 0)
        {
            std::cout << "***************************************" << std::endl;
            std::cout << "开始计算哈密顿量 P = " << P << " , 当前时间: " << get_current_time();
            std::cout << "子空间维数: " << dim << " ,每个进程任务数量: " << part << std::endl;
            std::cout << "任务进度: " << std::endl;
        }

        std::vector<Eigen::Triplet<complexd>> tripletList;
        createH(part_begin, part_end, tripletList);
        MPI_Barrier(MPI_COMM_WORLD);

        // 将结果汇集到主进程
        std::vector<Eigen::Triplet<complexd>> globalTripletList;
        if (rank == 0)
        {
            globalTripletList = tripletList;
            for (int i = 1; i < size; i++)
            {
                int recvSize;
                MPI_Recv(&recvSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<Eigen::Triplet<complexd>> recvTripletList(recvSize);
                MPI_Recv(recvTripletList.data(), recvSize * sizeof(Eigen::Triplet<complexd>), MPI_BYTE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                globalTripletList.insert(globalTripletList.end(), recvTripletList.begin(), recvTripletList.end());
            }
        }
        else
        {
            int sendSize = tripletList.size();
            MPI_Send(&sendSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(tripletList.data(), sendSize * sizeof(Eigen::Triplet<complexd>), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        }

        // 主进程保存矩阵到Hsave文件夹（总是保存，供Python程序使用）
        if (rank == 0)
        {
            std::cout << std::endl; // 换行，结束进度输出
            std::cout << "哈密顿量计算完成, 当前时间: " << get_current_time();
            std::cout << "正在保存矩阵到文件..." << std::endl;
            std::string filepath = "./Hsave/matrix_" + std::to_string(P) + ".h5";
            std::vector<int> rowIndices, colIndices;
            std::vector<complexd> values;
            for (const auto &triplet : globalTripletList)
            {
                rowIndices.push_back(triplet.row());
                colIndices.push_back(triplet.col());
                values.push_back(triplet.value());
            }
            saveMatrix(rowIndices, colIndices, values, filepath);
            std::cout << "矩阵保存完成." << std::endl;
            std::cout << "子空间 P = " << P << " 计算完成, 当前时间: " << get_current_time();
        }
    } // end for P

    // ============================== 调用Python程序进行对角化 ============================
    if (rank == 0)
    {
        std::cout << "\n"
                  << std::string(60, '=') << std::endl;
        std::cout << "所有哈密顿量计算完成！" << std::endl;

        if (ifDiagMatrix)
        {
            std::cout << "开始调用Python程序对角化..." << std::endl;
            std::cout << "当前时间: " << get_current_time();

            // 调用Python程序进行对角化（使用完整路径避免环境冲突）
            std::string python_cmd = "/public/home/baixing/miniconda3/bin/python diagmatrix.py";
            int result = system(python_cmd.c_str());

            if (result == 0)
            {
                std::cout << "Python对角化程序执行成功！" << std::endl;
            }
            else
            {
                std::cerr << "Python对角化程序执行失败，返回代码: " << result << std::endl;
            }
        }
        else
        {
            std::cout << "根据设置跳过对角化步骤。" << std::endl;
        }

        // 根据用户设置决定是否清空Hsave文件夹
        if (!ifSaveMatrix)
        {
            std::cout << "根据设置清空Hsave文件夹..." << std::endl;
            std::string cleanup_cmd = "rm -rf ./Hsave/*";
            system(cleanup_cmd.c_str());
            std::cout << "Hsave文件夹已清空。" << std::endl;
        }
        else
        {
            std::cout << "根据设置保留Hsave中的矩阵文件。" << std::endl;
        }

        std::cout << "程序完成，当前时间: " << get_current_time();
        std::cout << std::string(60, '=') << std::endl;
    }

    MPI_Finalize();
    return 0;
}

// =================================== 工具函数 ===================================
std::string get_current_time()
{
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    return std::ctime(&now_c);
}

/**
 * @brief Check if the folder exists, if not, create it; if exists, clean it.
 */
void checkFolder(const std::string &path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        std::string command = "mkdir -p " + path;
        system(command.c_str());
    }
    else
    {
        std::string command = "rm -rf " + path + "/*";
        system(command.c_str());
    }
}