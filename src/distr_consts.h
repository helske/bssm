// constants for particle filter
// use same notation as in bssm models

#ifndef CONST_H
#define CONST_H

#include <RcppArmadillo.h>
#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "nlg_ssm.h"

double norm_log_const(double sd);
double poisson_log_const(double y, double u);
double binomial_log_const(double y, double u);
double negbin_log_const(double y, double u, double phi);

double norm_log_const(const arma::vec& y, const arma::vec& sd);
double poisson_log_const(const arma::vec& y, const arma::vec& u);
double binomial_log_const(const arma::vec& y, const arma::vec& u);
double negbin_log_const(const arma::vec&  y, const arma::vec& u, double phi);
double compute_const_term(const ung_ssm& model, const ugg_ssm& approx_model);

#endif
