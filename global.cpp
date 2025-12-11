#include "global.h"

double U1, V1, U2, V2, W;
int Nsp1, Nsp2, Nsplit;
double gamma1, gamma2;
bool ifSaveMatrix, ifDiagMatrix;

int rank, size;
int P;
double H0[Ns] = {0};
complexd Psi0[Ns][Nsub] = {0};
std::vector<long long> subspace;