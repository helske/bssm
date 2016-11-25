#ifndef SAMPLE_H
#define SAMPLE_H

#include <RcppArmadillo.h>

arma::uvec stratified_sample(arma::vec p, arma::vec& r, unsigned int N);

#endif
