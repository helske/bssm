#ifndef MILSTEIN_FN_H
#define MILSTEIN_FN_H

#include <RcppArmadillo.h>
#include "sitmo.h"

// typedef for a pointer of drift/diffusion functions
typedef double (*fnPtr)(const double x, const arma::vec&);

// Functions for the Milstein scheme

// Calculates the terminal values of the Milstein discretisations of a
// SDE in [0,t] using 2^L levels
// In:
//  x0     -- Starting point
//  mu     -- Drift function
//  diffusion  -- Volatility function
//  ddiffusion -- Derivative of volatility
//  L      -- Discretisation level (2^L)
//  t      -- Terminal time
//  eng   -- RNG engine
// Out: list with entries
//  X      -- Terminal value

double milstein(const double x0, const unsigned int L, const double t,
  const arma::vec& theta,
  fnPtr drift, fnPtr diffusion, fnPtr ddiffusion,
  bool positive, sitmo::prng_engine& eng);

// A worker which uses simulated Brownian differences
double milstein_worker(double x, arma::vec& dB, double dt, unsigned int n,
  const arma::vec& theta, fnPtr drift, fnPtr diffusion,
  fnPtr ddiffusion, bool positive);



#endif
