#ifndef FN_PNTR_H
#define FN_PNTR_H

#include "bssm.h"

// typedef for a pointer of linear function of lgg-model equation returning vec (T, Z)
typedef arma::mat (*mat_fnPtr2)(const unsigned int t, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params);

// typedef for a pointer of nonlinear function of model equation returning vec (T, Z)
typedef arma::vec (*vec_fnPtr)(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params);
// typedef for a pointer of nonlinear function of model equation returning mat (Tg, Zg)
typedef arma::mat (*mat_fnPtr)(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params);
// typedef for a pointer of nonlinear function of model equation returning mat (R, H)
typedef arma::mat (*mat_varfnPtr)(const unsigned int t, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params);
// typedef for a pointer of nonlinear function of model equation returning vec (a1)
typedef arma::vec (*vec_initfnPtr)(const arma::vec& theta, const arma::vec& known_params);
// typedef for a pointer of nonlinear function of model equation returning mat (P1)
typedef arma::mat (*mat_initfnPtr)(const arma::vec& theta, const arma::vec& known_params);
// typedef for a pointer of log-prior function
typedef double (*double_fnPtr)(const arma::vec&);

class double_fn {
  
public:

  double_fn(SEXP xps) {
    Rcpp::XPtr<double_fnPtr> xptr(xps);
    funptr = *(xptr);
  }
  double eval(const arma::vec& theta) const {
    return funptr(theta);
  }
  
private:
  double_fnPtr funptr;
};

class vec_fn {
  
public:
  vec_fn(SEXP xps) {
    Rcpp::XPtr<vec_fnPtr> xptr(xps);
    funptr = *(xptr);
  }
  arma::vec eval(unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
    const arma::vec& known_params, const arma::mat& known_tv_params) const {
    return funptr(t, alpha, theta, known_params, known_tv_params);
  }
  
private:
  vec_fnPtr funptr;
};

class mat_fn {
  
public:
  mat_fn(SEXP xps) {
    Rcpp::XPtr<mat_fnPtr> xptr(xps);
    funptr = *(xptr);
  }
  arma::mat eval(unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
    const arma::vec& known_params, const arma::mat& known_tv_params) const {
    return funptr(t, alpha, theta, known_params, known_tv_params);
  }
  
private:
  mat_fnPtr funptr;
};

class mat_fn2 {
  
public:
  mat_fn2(SEXP xps) {
    Rcpp::XPtr<mat_fnPtr2> xptr(xps);
    funptr = *(xptr);
  }
  arma::mat eval(unsigned int t, const arma::vec& theta, 
    const arma::vec& known_params, const arma::mat& known_tv_params) const {
    return funptr(t, theta, known_params, known_tv_params);
  }
  
private:
  mat_fnPtr2 funptr;
};

class mat_varfn {
  
public:
  mat_varfn(SEXP xps) {
    Rcpp::XPtr<mat_varfnPtr> xptr(xps);
    funptr = *(xptr);
  }
  arma::mat eval(unsigned int t, const arma::vec& theta, 
    const arma::vec& known_params, const arma::mat& known_tv_params) const {
    return funptr(t, theta, known_params, known_tv_params);
  }
  
private:
  mat_varfnPtr funptr;
};

class vec_initfn {
  
public:
  
  vec_initfn(SEXP xps) {
    Rcpp::XPtr<vec_initfnPtr> xptr(xps);
    funptr = *(xptr);
  }
  arma::vec eval(const arma::vec& theta, const arma::vec& known_params) const {
    return funptr(theta, known_params);
  }
  
private:
  vec_initfnPtr funptr;
};

class mat_initfn {
  
public:
  mat_initfn(SEXP xps) {
    Rcpp::XPtr<mat_initfnPtr> xptr(xps);
    funptr = *(xptr);
  }
  arma::mat eval(const arma::vec& theta, const arma::vec& known_params) const {
    return funptr(theta, known_params);
  }
  
private:
  mat_initfnPtr funptr;
};

#endif