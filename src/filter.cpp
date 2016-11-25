#include <RcppArmadillo.h>
#include "bssm.h"

// stand-alone Kalman filter
double uv_filter_update(const double y, const arma::vec& Z, const double HH,
  arma::subview_col<double> at, arma::mat& Pt, arma::subview_col<double> att, arma::mat& Ptt, 
  const double zero_tol) {
  double logLik = 0;
  // update
  double F = arma::as_scalar(Z.t() * Pt * Z + HH);
  if (arma::is_finite(y) && F > zero_tol) {
    double v = arma::as_scalar(y - Z.t() * at);
    arma::vec K = Pt * Z / F;
    att = at + K * v;
    Ptt = Pt - K * K.t() * F;
    logLik = -0.5 * (LOG2PI + log(F) + v * v/F);
  } else {
    att = at;
    Ptt = Pt;
  }
  return logLik;
}
void uv_filter_predict(const arma::mat& T, const arma::mat& RR, const arma::vec& C, 
  arma::subview_col<double> att, arma::mat& Ptt, arma::subview_col<double> at, arma::mat& Pt) {

  // prediction
  at = C + T * att;
  Pt = arma::symmatu(T * Ptt * T.t() + RR);
}

// [[Rcpp::export]]
double uv_filter(const double y, const arma::vec& Z, const double HH,
  const arma::mat& T, const arma::mat& RR, const arma::vec& C, arma::vec& at, arma::mat& Pt, 
  const double zero_tol) {

  double logLik = 0;
  double F = arma::as_scalar(Z.t() * Pt * Z + HH);
  if (arma::is_finite(y) && F > zero_tol) {
    double v = arma::as_scalar(y - Z.t() * at);
    arma::vec K = Pt * Z / F;
    at = C + T * (at + K * v);
    Pt = arma::symmatu(T * (Pt - K * K.t() * F) * T.t() + RR);
    logLik = -0.5 * (LOG2PI + log(F) + v * v/F);
  } else {
    at = C + T * at;
    Pt = arma::symmatu(T * Pt * T.t() + RR);
  }
  return logLik;
}
