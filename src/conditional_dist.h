#ifndef CDIST_H
#define CDIST_H

#include "bssm.h"

void conditional_cov(arma::cube& V, arma::cube& C, const bool use_svd = true);

#endif
