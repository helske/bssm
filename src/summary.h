#ifndef SUMMARY_H
#define SUMMARY_H

#include <sitmo.h>
#include "bssm.h"
#include "filter_smoother.h"

void summary(
    const arma::cube& x, 
    arma::mat& mean_x, 
    arma::cube& cov_x);

void weighted_summary(
    const arma::cube& x, 
    arma::mat& mean_x,
    arma::cube& cov_x, 
    const arma::vec& weights);

void filter_summary(
    const arma::cube& alpha, 
    arma::mat& at, 
    arma::mat& att, 
    arma::cube& Pt, 
    arma::cube& Ptt, 
    arma::mat weights);

void sample_or_summarise(
    bool sample,
    const unsigned int method, 
    arma::cube& alpha, 
    const arma::vec& weights, 
    const arma::umat& indices,
    arma::mat& sampled_alpha, 
    arma::mat& alphahat, 
    arma::cube& Vt,  
    sitmo::prng_engine& engine);
#endif
