#ifndef SAMPLE_H
#define SAMPLE_H

#include <RcppArmadillo.h>

arma::uvec stratified_sample(arma::vec& p, const arma::vec& r, const unsigned int N);

#endif
