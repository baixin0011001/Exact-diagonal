#ifndef MODEL
#define MODEL

#include "global.h"
#include <Eigen/Sparse>

struct MatEle
{
    int newIndex;
    complexd value;

    MatEle(int idx = 0, complexd val = 0.0) : newIndex(idx), value(val) {}
};

void diagH0(int idx);
void V_new(long long Rstate, std::vector<MatEle> &mat);
void createH(int part_begin, int part_end, std::vector<Eigen::Triplet<complexd>> &tripletList);
void diagMatrix(const Eigen::SparseMatrix<complexd> &H, const std::string &filename);

#endif