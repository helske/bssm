#ifndef GSSM_H
#define GSSM_H

#include "bssm.h"

class gssm {

public:

  gssm(arma::vec, arma::mat, arma::vec, arma::cube, arma::cube, arma::vec,
    arma::mat, arma::mat, arma::vec, unsigned int);
  gssm(arma::vec, arma::mat, arma::vec, arma::cube, arma::cube, arma::vec,
    arma::mat, arma::mat, arma::vec, arma::uvec, arma::uvec, arma::uvec,
    arma::uvec, unsigned int);

  virtual double proposal(const arma::vec&, const arma::vec&);
  virtual void update_model(arma::vec);
  virtual arma::vec get_theta(void);
  virtual void compute_RR(void);
  virtual void compute_HH(void);
  virtual void compute_xbeta(void);
  virtual double log_likelihood(bool);
  virtual double filter(arma::mat&, arma::mat&, arma::cube&, arma::cube&, bool);
  virtual arma::mat fast_smoother(bool);
  virtual arma::mat fast_smoother2(arma::vec&, arma::mat&, arma::cube&, bool);
  virtual arma::mat precomp_fast_smoother(const arma::vec&, const arma::mat&,
    const arma::cube&, bool);
  virtual arma::cube sim_smoother(unsigned int, bool);
  virtual void smoother(arma::mat&, arma::cube&, bool);


  virtual arma::mat predict2(arma::vec, arma::vec, unsigned int, unsigned int,
    unsigned int, unsigned int, double, double, arma::mat, unsigned int,
    unsigned int);
  virtual List predict(arma::vec, arma::vec, unsigned int, unsigned int,
    unsigned int, double, double, arma::mat, unsigned int,
    unsigned int, arma::vec);

  virtual double mcmc_full(arma::vec, arma::vec, unsigned int, unsigned int,
    unsigned int, unsigned int, double, double, arma::mat&,
    arma::cube&, arma::mat&, arma::vec&, bool);

  virtual double mcmc_param(arma::vec, arma::vec, unsigned int,
    unsigned int, unsigned int, double, double, arma::mat&,
    arma::mat&, arma::vec&, bool);

  virtual double mcmc_param2(arma::vec, arma::vec, unsigned int,
    unsigned int, unsigned int, double, double, arma::mat&,
    arma::mat&, arma::vec&, arma::uvec&, bool);

  virtual double mcmc_summary(arma::vec, arma::vec, unsigned int, unsigned int,
    unsigned int, double, double, arma::mat&, arma::mat&, arma::cube&,
    arma::mat&, arma::vec&, bool);

  virtual arma::vec pyt(const unsigned int, const arma::cube&);
  virtual double bootstrap_filter(unsigned int, arma::cube&, arma::vec&);
  
  arma::vec y;
  arma::mat Z;
  arma::vec H;
  arma::cube T;
  arma::cube R;
  arma::cube Q;
  arma::vec a1;
  arma::mat P1;

  const unsigned int Ztv;
  const unsigned int Htv;
  const unsigned int Ttv;
  const unsigned int Rtv;

  const unsigned int n;
  const unsigned int m;
  const unsigned int k;

  arma::vec HH;
  arma::cube RR;

  const arma::mat xreg;
  arma::vec beta;
  arma::vec xbeta;

  arma::uvec Z_ind;
  arma::uvec H_ind;
  arma::uvec T_ind;
  arma::uvec R_ind;
  std::mt19937 engine;
  const double zero_tol;
};

#endif
