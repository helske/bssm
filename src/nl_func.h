#ifndef NL_FUNC_H
#define NL_FUNC_H

#include <RcppArmadillo.h>

// typedef for a pointer of nonlinear function of model equation
typedef arma::vec (*fnPtr)(const arma::vec&, const arma::vec&);
// typedef for a pointer of derivative of nonlinear function of model equation
typedef arma::mat (*gnPtr)(const arma::vec&, const arma::vec&);

class nonlinear_fn {
  
public:
  // eval(Rcpp::XPtr<funcPtr> xptr) {
  //   funptr = *(xptr);
  // }
  nonlinear_fn(SEXP xps) {
    Rcpp::XPtr<fnPtr> xptr(xps);
    funptr = *(xptr);
  }
  arma::vec eval(const arma::vec& alpha, const arma::vec& theta) const {
    return funptr(alpha, theta);
  }
  
private:
  fnPtr funptr;
};

class nonlinear_gn {
  
public:
  nonlinear_gn(SEXP xps) {
    Rcpp::XPtr<gnPtr> xptr(xps);
    funptr = *(xptr);
  }
  arma::mat eval(const arma::vec& alpha, const arma::vec& theta) const {
    return funptr(alpha, theta);
  }
  
private:
  gnPtr funptr;
};

#endif