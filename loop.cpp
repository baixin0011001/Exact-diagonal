#include "model.h"
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

void createH(int part_begin, int part_end, std::vector<Eigen::Triplet<complexd>> &tripletList)
{
    for (int i = part_begin; i < part_end; i++)
    {
        if (rank == 0 && (i + 1) % 500 == 0)
        {
            std::cout << i + 1 << ", ";
        }
        std::vector<MatEle> mat;
        // 动能项
        double temp = 0.0;
        for (int k = 0; k < Ns; k++)
        {
            if ((subspace[i] >> k) & 1)
                temp += H0[k];
        }
        mat.emplace_back(i, temp);

        // 相互作用项
        V_new(subspace[i], mat);

        // 将矩阵元添加到tripletList中
        for (const auto &ele : mat)
        {
            if (fabs(ele.value.real()) < 1e-6 && fabs(ele.value.imag()) < 1e-6)
                continue; // 如果矩阵元过小，则忽略
            tripletList.emplace_back(ele.newIndex, i, ele.value);
        }
    }
}

void diagMatrix(const Eigen::SparseMatrix<complexd> &H, const std::string &filename)
{
    int ne = 1;    // 需要计算的本征值个数
    int ncv = 500; // Arnoldi方法的近似维数
    Spectra::SparseSymMatProd<complexd> op(H);
    Spectra::SymEigsSolver<Spectra::SparseSymMatProd<complexd>> eigs(op, ne, ncv);
    eigs.init();
    int nconv = eigs.compute(Spectra::SortRule::SmallestAlge);

    // 检查结果
    if (eigs.info() == Spectra::CompInfo::Successful)
    {
        Eigen::VectorXd eigenvalues = eigs.eigenvalues();
        Eigen::MatrixXcd eigenvectors = eigs.eigenvectors();
        // 将本征值转换为 std::vector<double>
        std::vector<double> eigenvaluesVec(eigenvalues.data(), eigenvalues.data() + eigenvalues.size());
        // 将本征向量转换为 std::vector<std::vector<complexd>>
        std::vector<std::vector<complexd>> eigenvectorsVec;
        for (int i = 0; i < eigenvectors.cols(); i++)
        {
            std::vector<complexd> vec(eigenvectors.col(i).data(), eigenvectors.col(i).data() + eigenvectors.rows());
            eigenvectorsVec.push_back(vec);
        }

        // 保存基态信息（只保存最小的本征值和对应的本征矢）
        appendGroundState(eigenvaluesVec[0], eigenvectorsVec[0], P);
    }
    else
    {
        std::cerr << "Eigenvalue computation failed!" << std::endl;
        if (eigs.info() == Spectra::CompInfo::NotConverging)
            std::cerr << "Not converging. Try increasing the number of Arnoldi vectors." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}