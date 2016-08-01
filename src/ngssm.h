#ifndef NGSSM_H
#define NGSSM_H

#include "gssm.h"

class ngssm: public gssm {
public:

  ngssm(arma::vec, arma::mat, arma::cube, arma::cube, arma::vec,
    arma::mat, arma::vec, arma::mat, arma::vec, unsigned int, unsigned int);
  ngssm(arma::vec, arma::mat, arma::cube, arma::cube, arma::vec,
    arma::mat, arma::vec, arma::mat, arma::vec, unsigned int, arma::uvec,
    arma::uvec, arma::uvec, unsigned int);

  virtual double proposal(const arma::vec&, const arma::vec&);
  virtual void update_model(arma::vec);
  virtual arma::vec get_theta(void);
  virtual double approx(arma::vec&, unsigned int, double);
  virtual double logp_signal(arma::vec&, arma::mat&, arma::vec&);
  virtual double precomp_logp_signal(arma::vec&, const arma::mat&, const arma::vec&);
  virtual double logp_y(arma::vec&);
  virtual arma::vec approx_iter(arma::vec&);

  virtual arma::mat predict2(arma::vec, arma::vec, unsigned int, unsigned int,
    unsigned int, unsigned int, double, double, arma::mat, unsigned int,
    unsigned int, arma::vec);

  virtual arma::vec importance_weights(const arma::cube&);
  virtual double scaling_factor(const arma::vec&);


  virtual List mcmc_full(arma::vec, arma::vec, unsigned int,
    unsigned int, unsigned int, unsigned int, double, double, arma::mat, const arma::vec, bool);
  List mcmc_da(arma::vec, arma::vec, unsigned int,
    unsigned int, unsigned int, unsigned int, double, double, arma::mat, arma::vec, bool);

  double mcmc_approx(arma::vec, arma::vec,
    unsigned int, unsigned int, unsigned int,
    unsigned int, double, double, arma::mat&,
    const arma::vec, arma::mat&, arma::vec&,
    arma::mat&, arma::mat&, arma::vec&, bool);

  double mcmc_approx2(arma::vec, arma::vec,
      unsigned int, unsigned int, unsigned int,
      unsigned int, double, double, arma::mat&,
      const arma::vec, arma::mat&, arma::vec&,
      arma::mat&, arma::mat&, arma::vec&, arma::uvec&, bool);

  List mcmc_param(arma::vec, arma::vec,
    unsigned int, unsigned int, unsigned int,
    unsigned int, double, double, arma::mat,
    const arma::vec, bool);

  List mcmc_da_param(arma::vec, arma::vec, unsigned int,
    unsigned int, unsigned int, unsigned int, double, double, arma::mat, arma::vec, bool);

  List mcmc_summary(arma::vec, arma::vec,
    unsigned int, unsigned int, unsigned int,
    unsigned int, double, double, arma::mat,
    const arma::vec, bool);

  List mcmc_da_summary(arma::vec, arma::vec, unsigned int,
    unsigned int, unsigned int, unsigned int, double, double, arma::mat, arma::vec, bool);


  arma::cube invlink(const arma::cube&);

  arma::vec phi;
  unsigned int distribution;
  const arma::vec ng_y;
  unsigned int max_iter;
  double conv_tol;
};



#endif
