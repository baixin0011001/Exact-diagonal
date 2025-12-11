#include <Eigen/Dense>
#include "model.h"

using namespace Eigen;

int searchIndex(long long state);
void insertOne(std::vector<MatEle> &mat, const MatEle &ele);

/**
 * @brief 对角化单体哈密顿量, 保存第idx个本征值和对应的本征态。
 */
void diagH0(int idx)
{
    for (int q = 0; q < Ns; q++)
    {
        Matrix<complexd, Nsub, Nsub> h = Matrix<complexd, Nsub, Nsub>::Zero();

        double kx = (q % Nx) * 2 * PI / Nx + gamma1 / Nx;
        double ky = (q / Nx) * 2 * PI / Ny + gamma2 / Ny;

        // 第一层
        h(0, 0) = tp1 * (exp(-I * kx) + exp(I * kx)) + tp2 * (exp(-I * ky) + exp(I * ky)) +
                  tpp * (exp(-I * (kx + ky)) + exp(-I * (kx - ky)) + exp(I * (kx - ky)) + exp(I * (kx + ky)));
        h(0, 1) = t * (exp(-I * phi) + exp(I * phi) * exp(-I * kx) + exp(-I * phi) * exp(-I * (kx - ky)) + exp(I * phi) * exp(I * ky));
        h(1, 0) = t * (exp(I * phi) + exp(-I * phi) * exp(I * kx) + exp(I * phi) * exp(I * (kx - ky)) + exp(-I * phi) * exp(-I * ky));
        h(1, 1) = tp2 * (exp(-I * kx) + exp(I * kx)) + tp1 * (exp(-I * ky) + exp(I * ky)) +
                  tpp * (exp(-I * (kx + ky)) + exp(-I * (kx - ky)) + exp(I * (kx - ky)) + exp(I * (kx + ky)));

        // 第一层
        h(2, 2) = h(0, 0);
        h(2, 3) = h(0, 1);
        h(3, 2) = h(1, 0);
        h(3, 3) = h(1, 1);

        // 层间跃迁
        h(0, 3) = tbot;
        h(3, 0) = tbot;
        // h(1, 2) = tbot;
        // h(2, 1) = tbot;

        SelfAdjointEigenSolver<Matrix<complexd, Nsub, Nsub>> solver(h);
        Vector<double, Nsub> eigenvalues = solver.eigenvalues();
        Matrix<complexd, Nsub, Nsub> eigenvectors = solver.eigenvectors();

        H0[q] = eigenvalues[idx];
        for (int i = 0; i < Nsub; i++)
        {
            Psi0[q][i] = eigenvectors(i, idx);
        }
    }
}

std::vector<std::vector<complexd>> V_old(int q1x, int q1y, int q4x, int q4y)
{
    Matrix<complexd, Nsub, Nsub> v = Matrix<complexd, Nsub, Nsub>::Zero();

    double k1x = 2.0 * PI * q1x / Nx;
    double k1y = 2.0 * PI * q1y / Ny;
    double k4x = 2.0 * PI * q4x / Nx;
    double k4y = 2.0 * PI * q4y / Ny;

    // 第一层
    v(0, 0) = 1. / Nu * V1 * (exp(-I * (k1x - k4x)) + exp(-I * (k1y - k4y)) + exp(I * (k1x - k4x)) + exp(I * (k1y - k4y)));
    v(0, 1) = 1. / Nu * U1 * (1.0 + exp(-I * (k1x - k4x)) + exp(-I * (k1x - k4x - k1y + k4y)) + exp(I * (k1y - k4y)));
    v(1, 0) = 1. / Nu * U1 * (1.0 + exp(I * (k1x - k4x)) + exp(I * (k1x - k4x - k1y + k4y)) + exp(-I * (k1y - k4y)));
    v(1, 1) = 1. / Nu * V1 * (exp(-I * (k1x - k4x)) + exp(-I * (k1y - k4y)) + exp(I * (k1x - k4x)) + exp(I * (k1y - k4y)));

    // 第二层
    v(2, 2) = 1. / Nu * V2 * (exp(-I * (k1x - k4x)) + exp(-I * (k1y - k4y)) + exp(I * (k1x - k4x)) + exp(I * (k1y - k4y)));
    v(2, 3) = 1. / Nu * U2 * (1.0 + exp(-I * (k1x - k4x)) + exp(-I * (k1x - k4x - k1y + k4y)) + exp(I * (k1y - k4y)));
    v(3, 2) = 1. / Nu * U2 * (1.0 + exp(I * (k1x - k4x)) + exp(I * (k1x - k4x - k1y + k4y)) + exp(-I * (k1y - k4y)));
    v(3, 3) = 1. / Nu * V2 * (exp(-I * (k1x - k4x)) + exp(-I * (k1y - k4y)) + exp(I * (k1x - k4x)) + exp(I * (k1y - k4y)));

    // 层间相互作用
    v(0, 3) = 1.0 / Nu * W;
    v(3, 0) = 1.0 / Nu * W;

    std::vector<std::vector<std::complex<double>>> result;
    for (int i = 0; i < v.rows(); i++)
    {
        std::vector<std::complex<double>> row;
        for (int j = 0; j < v.cols(); j++)
        {
            row.push_back(v(i, j));
        }
        result.push_back(row);
    }
    return result;
}

complexd Vkbar(long long k1, long long k2, long long k3, long long k4)
{
    std::vector<std::vector<complexd>> v_old = V_old(k1 % Nx, k1 / Nx, k4 % Nx, k4 / Nx);
    complexd v_tmp(0., 0.);
    for (int i = 0; i < Nsub; i++)
    {
        for (int j = 0; j < Nsub; j++)
        {
            v_tmp += conj(Psi0[k1][i]) * conj(Psi0[k2][j]) * Psi0[k3][j] * Psi0[k4][i] * v_old[i][j];
        }
    }
    return v_tmp;
}

void V_new(const long long Rstate, std::vector<MatEle> &mat)
{
    std::vector<long long> popu; // 占据的位置
    // std::vector<int> notpopu; // 未占据的位置
    for (long long k = 0; k < Ns; k++)
    {
        if ((Rstate >> k) & 1)
            popu.push_back(k);
        // else
        //     notpopu.push_back(k);
    }

    // 将产生湮灭算符作用上去
    long long newState;
    double sign;
    for (long long k4 = 0; k4 < N; k4++)
    {
        int k4x = popu[k4] % Nx;
        int k4y = popu[k4] / Nx;

        for (long long k3 = 0; k3 < N; k3++)
        {
            if (k3 == k4)
                continue;
            // std::vector<int> notpopu_backup = notpopu;
            int k3x = popu[k3] % Nx;
            int k3y = popu[k3] / Nx;

            std::vector<long long> notpopu_backup;
            for (long long k = 0; k < Ns; k++)
            {
                if (((Rstate >> k) & 1) == 0)
                    notpopu_backup.push_back(k);
            }
            notpopu_backup.push_back(popu[k4]);
            notpopu_backup.push_back(popu[k3]);

            for (long long k2 = 0; k2 < notpopu_backup.size(); k2++)
            {
                int k2x = notpopu_backup[k2] % Nx;
                int k2y = notpopu_backup[k2] / Nx;

                for (long long k1 = 0; k1 < notpopu_backup.size(); k1++)
                {
                    if (k1 == k2)
                        continue;
                    int k1x = notpopu_backup[k1] % Nx;
                    int k1y = notpopu_backup[k1] / Nx;

                    // 满足动量守恒
                    if (((k3x + k4x - k1x - k2x) % Nx != 0) or ((k3y + k4y - k1y - k2y) % Ny != 0))
                        continue;

                    newState = (Rstate ^ (1LL << popu[k4])) ^ (1LL << popu[k3]); // 湮灭算符用异或表示
                    if (k4 > k3)
                        sign = pow(-1., k4 - k3 - 1.);
                    else
                        sign = pow(-1., (double)k3 - k4);

                    int count = 0;
                    newState = newState ^ (1LL << notpopu_backup[k2]); // 产生算符也用异或表示
                    for (int j = notpopu_backup[k2] + 1; j < Ns; j++)
                        if (newState & (1LL << j))
                            count++;
                    newState = newState ^ (1LL << notpopu_backup[k1]);
                    for (int j = notpopu_backup[k1] + 1; j < Ns; j++)
                        if (newState & (1LL << j))
                            count++;

                    sign *= pow(-1., (double)count);

                    complexd value = sign * Vkbar(notpopu_backup[k1], notpopu_backup[k2], popu[k3], popu[k4]);
                    if (fabs(value.real()) < 1e-6 && fabs(value.imag()) < 1e-6)
                        continue;                         // 如果矩阵元过小，则不考虑
                    int newIndex = searchIndex(newState); // 新态的索引
                    if (newIndex == -1)
                    {
                        std::cerr << "Error: new state not found in subspace for state " << newState << std::endl;
                        MPI_Finalize();
                        exit(1);
                    }
                    MatEle ele(newIndex, value);
                    insertOne(mat, ele);

                } // end for k1
            } // end for k2

        } // end for k3
    } // end for k4
}
// ===================================== 辅助函数 =====================================
/**
 * @brief 搜索给定态在子空间中的索引。
 */
int searchIndex(long long state)
{
    int left = 0;
    int right = subspace.size() - 1;

    while (left <= right)
    {
        int mid = left + (right - left) / 2;

        if (subspace[mid] == state)
            return mid;
        else if (subspace[mid] < state)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}

/**
 * @brief 向矩阵元列表中插入一个新的矩阵元。
 */
void insertOne(std::vector<MatEle> &mat, const MatEle &ele)
{
    int left = 0;
    int right = mat.size() - 1;

    while (left <= right)
    {
        int mid = left + (right - left) / 2;
        if (mat[mid].newIndex == ele.newIndex)
        {
            mat[mid].value += ele.value; // 如果已经存在，则累加值
            return;
        }
        else if (mat[mid].newIndex < ele.newIndex)
            left = mid + 1;
        else
            right = mid - 1;
    }

    mat.insert(mat.begin() + left, ele);
}