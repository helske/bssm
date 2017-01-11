#ifndef GSSM_H
#define GSSM_H

#include <RcppArmadillo.h>

class gssm {

public:

  gssm(const Rcpp::List&, unsigned int);
  gssm(const Rcpp::List&, const arma::uvec&, const arma::uvec&, const arma::uvec&, 
    const arma::uvec&, unsigned int);
  gssm(const Rcpp::List&, unsigned int, bool);
  gssm(const Rcpp::List&, const arma::uvec&, const arma::uvec&, 
    const arma::uvec&, unsigned int, bool);

  void compute_RR();
  void compute_HH();
  void compute_xbeta();
  
  double log_likelihood(bool);
  double filter(arma::mat&, arma::mat&, arma::cube&, arma::cube&, bool);
  arma::mat fast_smoother(bool);
  arma::mat fast_smoother2(arma::vec&, arma::mat&, arma::cube&, bool);
  arma::mat precomp_fast_smoother(const arma::vec&, const arma::mat&,
    const arma::cube&, bool);
  arma::cube sim_smoother(unsigned int, bool);
  void smoother(arma::mat&, arma::cube&, bool);
  void smoother_ccov(arma::mat&, arma::cube&, arma::cube&, bool);
  
  virtual double proposal(const arma::vec&, const arma::vec&);
  virtual void update_model(arma::vec);
  virtual arma::vec get_theta();
  virtual double prior_pdf(const arma::vec&, const arma::uvec&, const arma::mat&);

  virtual arma::mat predict2(const arma::uvec&, const arma::mat&, unsigned int, unsigned int,
    unsigned int, double, double, arma::mat, unsigned int,
    unsigned int);
  virtual Rcpp::List predict(const arma::uvec&, const arma::mat&, unsigned int, unsigned int,
    unsigned int, double, double, arma::mat, unsigned int,
    unsigned int, arma::vec);

  virtual double run_mcmc(const arma::uvec&, const arma::mat&, unsigned int, bool,
    unsigned int, unsigned int, double, double, arma::mat&, bool,
    arma::mat&, arma::vec&, arma::cube&);

  virtual double mcmc_summary(const arma::uvec&, const arma::mat&, unsigned int, unsigned int,
    unsigned int, double, double, arma::mat&, bool, arma::mat&, arma::vec&, arma::mat&, arma::cube&);

  virtual double particle_filter(unsigned int, arma::cube&, arma::mat&, arma::umat&);
  virtual double psi_filter(unsigned int, arma::cube&, arma::mat&, arma::umat&);
  
  void backtrack_pf2(const arma::cube&, arma::mat&, const arma::umat&);

  arma::mat backward_simulate(arma::cube&, arma::mat&, arma::umat&);

  arma::vec y;
  arma::mat Z;
  arma::vec H;
  arma::cube T;
  arma::cube R;
  arma::cube Q;
  arma::vec a1;
  arma::mat P1;
  arma::mat xreg;
  arma::vec beta;
  arma::mat C;
  
  const unsigned int Ztv;
  const unsigned int Htv;
  const unsigned int Ttv;
  const unsigned int Rtv;
  const unsigned int Ctv;

  const unsigned int n;
  const unsigned int m;
  const unsigned int k;

  arma::vec HH;
  arma::cube RR;

  arma::vec xbeta;

  arma::uvec Z_ind;
  arma::uvec H_ind;
  arma::uvec T_ind;
  arma::uvec R_ind;
  std::mt19937 engine;
  const double zero_tol;
  
  const arma::vec noise_const;
};


#endif
