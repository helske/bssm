#include "bssm.h"
// stand-alone Kalman filter
double uv_filter_update(const double y, arma::subview_col<double> Z, const double HH,
  arma::subview_col<double> at, arma::mat& Pt, arma::subview_col<double> att, arma::mat& Ptt, const double zero_tol) {
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
void uv_filter_predict(const arma::mat& T, const arma::mat& RR, arma::subview_col<double> att,
  arma::mat& Ptt, arma::subview_col<double> at, arma::mat& Pt) {

  // prediction
  at = T * att;
  Pt = arma::symmatu(T * Ptt * T.t() + RR);
}


double uv_filter(const double y, const arma::vec& Z, const double HH,
  const arma::mat& T, const arma::mat& RR, arma::vec& at, arma::mat& Pt, const double zero_tol) {

  double logLik = 0;
  double F = arma::as_scalar(Z.t() * Pt * Z + HH);
  if (arma::is_finite(y) && F > zero_tol) {
    double v = arma::as_scalar(y - Z.t() * at);
    arma::vec K = Pt * Z / F;
    at = T * (at + K * v);
    Pt = arma::symmatu(T * (Pt - K * K.t() * F) * T.t() + RR);
    logLik = -0.5 * (LOG2PI + log(F) + v * v/F);
  } else {
    at = T * at;
    Pt = arma::symmatu(T * Pt * T.t() + RR);
  }
  return logLik;
}

//for univariate particle filter
void uv_filter2(const double y, const double HH, const double T, const double RR,
  double& at, double& Pt, const double zero_tol) {
 
  // prediction
    at = T * at;
    Pt = RR;
  // update
  if (arma::is_finite(y)) {
    double F = Pt + HH;
    double v = y - at;
    double K = Pt / F;
    at += K * v;
    Pt = (1.0 - K)*Pt;
  }
}