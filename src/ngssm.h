#ifndef NGSSM_H
#define NGSSM_H

#include "gssm.h"

class ngssm: public gssm {
public:
  
  
  ngssm(const Rcpp::List&, unsigned int);
  ngssm(const Rcpp::List&, arma::uvec, arma::uvec, arma::uvec, unsigned int);
  
  ngssm(arma::vec, arma::mat, arma::cube, arma::cube, arma::vec,
    arma::mat, double, arma::vec, arma::mat, arma::vec, arma::mat, unsigned int, unsigned int, bool);
  ngssm(arma::vec, arma::mat, arma::cube, arma::cube, arma::vec,
    arma::mat, double, arma::vec, arma::mat, arma::vec, arma::mat, unsigned int, arma::uvec,
    arma::uvec, arma::uvec, unsigned int, bool);
  
  virtual double proposal(const arma::vec&, const arma::vec&);
  virtual void update_model(arma::vec);
  virtual arma::vec get_theta(void);
  virtual double approx(arma::vec&, unsigned int, double);
  virtual double logp_signal(arma::vec&, arma::mat&, arma::vec&);
  virtual double precomp_logp_signal(arma::vec&, const arma::mat&, const arma::vec&);
  virtual double logp_y(arma::vec&);
  virtual arma::vec approx_iter(arma::vec&);
  
  virtual arma::mat predict2(const arma::uvec&, const arma::mat&, unsigned int, unsigned int,
    unsigned int, unsigned int, double, double, arma::mat, unsigned int,
    unsigned int, arma::vec);
  
  arma::vec importance_weights(const arma::cube&);
  arma::vec importance_weights_t(const unsigned int, const arma::cube&);
  arma::vec importance_weights_t(const unsigned int, const arma::mat&);
  double scaling_factor(const arma::vec&);
  arma::vec scaling_factor_vec(const arma::vec&);
  double psi_loglik(unsigned int, const double, const arma::vec&);
  double bsf_loglik(unsigned int);
  double run_mcmc(const arma::uvec&, const arma::mat&, unsigned int,
    unsigned int, unsigned int, unsigned int, double, double, arma::mat&, const arma::vec,
    bool, bool, bool, arma::mat&, arma::vec&, arma::cube&);
  
  
  double run_mcmc_pf(const arma::uvec&, const arma::mat&, unsigned int,
    unsigned int, unsigned int, unsigned int, double, double, arma::mat&, const arma::vec,
    bool, bool, bool, arma::mat&, arma::vec&, arma::cube&, bool);
  
  double run_mcmc_summary(const arma::uvec&, const arma::mat&, unsigned int,
    unsigned int, unsigned int, unsigned int, double, double, arma::mat&, const arma::vec,
    bool, bool, bool, arma::mat&, arma::vec&, arma::mat&, arma::cube&, arma::mat&, arma::cube&);
  
  double mcmc_approx(const arma::uvec&, const arma::mat&,
    unsigned int, unsigned int, unsigned int,
    unsigned int, double, double, arma::mat&,
    const arma::vec, arma::mat&, arma::vec&, arma::vec&,
    arma::mat&, arma::mat&, arma::mat&, arma::uvec&, bool, bool);
  
  
  arma::cube invlink(const arma::cube&);
  
  arma::vec pyt(const unsigned int, const arma::cube&);
  arma::vec pyt(const unsigned int, const arma::mat&);
  
  double particle_filter(unsigned int, arma::cube&, arma::mat&, arma::umat&);
  double psi_filter(unsigned int, arma::cube&, arma::mat&, arma::umat&,
    const double, const arma::vec&);
  
  double phi;
  arma::vec ut;
  unsigned int distribution;
  bool phi_est;
  const arma::vec ng_y;
  unsigned int max_iter;
  double conv_tol;
};



#endif
