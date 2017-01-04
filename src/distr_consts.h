// constants for particle filter
// use same notation as in bssm models

#ifndef CONST_H
#define CONST_H

#include <RcppArmadillo.h>

double norm_log_const(double sd);
double poisson_log_const(double y, double u);
double binomial_log_const(double y, double u);
double negbin_log_const(double y, double u, double phi);

double poisson_log_const(const arma::vec& y, const arma::vec& u);
double binomial_log_const(const arma::vec& y, const arma::vec& u);
double negbin_log_const(const arma::vec&  y, const arma::vec& u, double phi);

#endif
