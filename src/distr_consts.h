#ifndef CONST_H
#define CONST_H

#include <RcppArmadillo.h>
// constants for particle filter
// use same notation as in bssm models

double norm_log_const(const double sd);
double poisson_log_const(const double y, const double u);
double binomial_log_const(const double y, const double u);
double negbin_log_const(const double y, const double u, const double phi);

double poisson_log_const(const arma::vec& y, const arma::vec& u);
double binomial_log_const(const arma::vec& y, const arma::vec& u);
double negbin_log_const(const arma::vec&  y, const arma::vec& u, const double phi);

#endif
