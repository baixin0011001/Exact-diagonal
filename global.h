#ifndef GLOBAL_H
#define GLOBAL_H

#include <mpi.h>
#include <complex>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using complexd = std::complex<double>;

constexpr double PI = 3.141592653589793;
constexpr complexd I(0.0, 1.0); // 虚数单位

// ===================================== 固定参数 =================================

constexpr int Nsub = 4; // 元胞内格点数
constexpr double t = -1.0;

const double tp1 = -0.2830; //-0.2681;  // 1.0/(2.0+sqrt(2.0));
const double tp2 = -tp1;    //-1./(2.+sqrt(2.));
const double tpp = 0.1884;  // 0.1601;  //      1./(2.+2.*sqrt(2.));
const double phi = 0.7868;  // 0.7851;  //      PI/4;
const double tbot = 0.5;

constexpr int Nx = 4;
constexpr int Ny = 6;
constexpr int Ns = Nx * Ny;
constexpr int Nu = Nx * Ny;
constexpr int N = 8; // 电子数

// ===================================== 输入参数 =================================
// 相互作用项
extern double U1; // 第一层最近邻
extern double V1; // 第一层次近邻
extern double U2; // 第二层最近邻
extern double V2; // 第二层最近邻
extern double W;  // 层间相互作用

extern int Nsp1;      // 扭转边界条件，步数，x方向
extern int Nsp2;      // 扭转边界条件，步数，y方向
extern int Nsplit;    // 扭转边界条件，总份数
extern double gamma1; // 第一倒格矢扭转角度
extern double gamma2; // 第二倒格矢扭转角度

extern bool ifSaveMatrix; // 是否保存矩阵
extern bool ifDiagMatrix; // 是否对矩阵对角化

// ===================================== 公共变量 =================================

extern int rank, size;
extern int P;                           // 总动量
extern double H0[Ns];                   // 自由部分能量本征值
extern complexd Psi0[Ns][Nsub];         // 自由部分能量本征态
extern std::vector<long long> subspace; // 子空间索引 (从小到大排序)

// ==================================== io中的函数 =================================

void readInputParameters();
std::vector<long long> loadSubspace();
void saveMatrix(const std::vector<int> &rows, const std::vector<int> &cols, const std::vector<complexd> &values, const std::string &filename);
void saveEigen(const std::vector<double> &eigenvalues, const std::vector<std::vector<complexd>> &eigenvectors, const std::string &filename);
void appendGroundState(double groundEnergy, const std::vector<complexd> &groundVector, int P);

#endif