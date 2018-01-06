#include "nlg_ssm.h"
#include "mgg_ssm.h"
#include "sample.h"
#include "dmvnorm.h"
#include "conditional_dist.h"
#include "rep_mat.h"
#include "psd_chol.h"
#include "interval.h"

nlg_ssm::nlg_ssm(const arma::mat& y, nvec_fnPtr Z_fn_, nmat_fnPtr H_fn_, nvec_fnPtr T_fn_, 
  nmat_fnPtr R_fn_, nmat_fnPtr Z_gn_, nmat_fnPtr T_gn_, a1_fnPtr a1_fn_, P1_fnPtr P1_fn_,
  const arma::vec& theta, prior_fnPtr log_prior_pdf_, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int m, const unsigned int k,
  const arma::uvec& time_varying, const unsigned int seed) :
  y(y), Z_fn(Z_fn_), H_fn(H_fn_), T_fn(T_fn_), 
  R_fn(R_fn_), Z_gn(Z_gn_), T_gn(T_gn_),
  a1_fn(a1_fn_), P1_fn(P1_fn_), theta(theta), 
  log_prior_pdf(log_prior_pdf_), known_params(known_params), 
  known_tv_params(known_tv_params), m(m), k(k), n(y.n_cols),  p(y.n_rows),
  Zgtv(time_varying(0)), Tgtv(time_varying(1)), Htv(time_varying(2)),
  Rtv(time_varying(3)), seed(seed), 
  engine(seed), zero_tol(1e-8) {
}

Rcpp::List nlg_ssm::predict_interval(const arma::vec& probs, const arma::mat& thetasim,
  const arma::mat& alpha_last, const arma::cube& P_last, 
  const arma::uvec& counts, const unsigned int predict_type) {
  
  if(p > 1) 
    Rcpp::stop("Interval prediction using EKF is currently not supported for multivariate observations.");
  theta = thetasim.col(0);
  
  arma::mat at(m, n);
  arma::cube Pt(m, m, n);
  at.col(0) = alpha_last.col(0);
  Pt.slice(0) = P_last.slice(0);
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    at.col(t + 1) = T_fn(t, at.col(t), theta, known_params, known_tv_params);
    arma::mat Tg = T_gn(t, at.col(t), theta, known_params, known_tv_params);
    arma::mat Rt = R_fn(t, at.col(t), theta, known_params, known_tv_params);
    Pt.slice(t + 1) = Tg * Pt.slice(t) * Tg.t() + Rt * Rt.t();
  }
  
  unsigned int n_samples = thetasim.n_cols;
  
  if (predict_type < 3) {
    
    arma::mat mean_pred(n, n_samples);
    arma::mat var_pred(n, n_samples);
    
    for(unsigned int t = 0; t < n; t++) {
      mean_pred(t, 0) = arma::as_scalar(Z_fn(t, at.col(t), theta, known_params, known_tv_params));
      arma::mat Zg = Z_gn(t, at.col(t), theta, known_params, known_tv_params);
      var_pred(t, 0) = arma::as_scalar(Zg * Pt.slice(t) * Zg.t());
    }
    
    if (predict_type == 1) {
      for(unsigned int t = 0; t < n; t++) {
        arma::mat HHt = H_fn(t, at.col(t), theta, known_params, known_tv_params);
        var_pred(t, 0) += arma::as_scalar(HHt * HHt.t());
      }
    }
    
    for (unsigned int i = 1; i < n_samples; i++) {
      
      theta = thetasim.col(i);
      at.col(0) = alpha_last.col(i);
      Pt.slice(0) = P_last.slice(i);
      
      for (unsigned int t = 0; t < (n - 1); t++) {
        at.col(t + 1) = T_fn(t, at.col(t), theta, known_params, known_tv_params);
        arma::mat Tg = T_gn(t, at.col(t), theta, known_params, known_tv_params);
        arma::mat Rt = R_fn(t, at.col(t), theta, known_params, known_tv_params);
        Pt.slice(t + 1) = Tg * Pt.slice(t) * Tg.t() + Rt * Rt.t();
      }
      for(unsigned int t = 0; t < n; t++) {
        mean_pred(t, i) = arma::as_scalar(Z_fn(t, at.col(t), theta, known_params, known_tv_params));
        arma::mat Zg = Z_gn(t, at.col(t), theta, known_params, known_tv_params);
        var_pred(t, i) = arma::as_scalar(Zg * Pt.slice(t) * Zg.t());
      }
      
      if (predict_type == 1) {
        for(unsigned int t = 0; t < n; t++) {
          arma::mat HHt = H_fn(t, at.col(t), theta, known_params, known_tv_params);
          var_pred(t, i) += arma::as_scalar(HHt * HHt.t());
        }
      }
      
    }
    
    arma::mat expanded_sd = rep_mat(arma::sqrt(var_pred), counts);
    arma::inplace_trans(expanded_sd);
    arma::mat expanded_mean = rep_mat(mean_pred, counts);
    arma::inplace_trans(expanded_mean);
    
    arma::mat intv = intervals(expanded_mean, expanded_sd, probs, n);
    
    return Rcpp::List::create(Rcpp::Named("intervals") = intv,
      Rcpp::Named("mean_pred") = expanded_mean,
      Rcpp::Named("sd_pred") = expanded_sd);
  } else {
    
    arma::cube mean_pred(n, n_samples, m);
    arma::cube var_pred(n, n_samples, m);
    
    for(unsigned int t = 0; t < n; t++) {
      mean_pred.tube(t, 0) = at.col(t);
      var_pred.tube(t, 0) = Pt.slice(t).diag();
    }
    
    for (unsigned int i = 1; i < n_samples; i++) {
      
      theta = thetasim.col(i);
      at.col(0) = alpha_last.col(i);
      Pt.slice(0) = P_last.slice(i);
      
      for (unsigned int t = 0; t < (n - 1); t++) {
        at.col(t + 1) = T_fn(t, at.col(t), theta, known_params, known_tv_params);
        arma::mat Tg = T_gn(t, at.col(t), theta, known_params, known_tv_params);
        arma::mat Rt = R_fn(t, at.col(t), theta, known_params, known_tv_params);
        Pt.slice(t + 1) = Tg * Pt.slice(t) * Tg.t() + Rt * Rt.t();
      }
      
      
      for(unsigned int t = 0; t < n; t++) {
        mean_pred.tube(t, i) = at.col(t);
        var_pred.tube(t, i) = Pt.slice(t).diag();
      }
    }
    
    arma::cube intv(n, probs.n_elem, m);
    arma::cube expanded_sd(arma::accu(counts), n, m);
    arma::cube expanded_mean(arma::accu(counts), n, m);
    for (unsigned int i = 0; i < m; i++) {
      arma::mat tmp1 = rep_mat(arma::sqrt(var_pred.slice(i)), counts);
      expanded_sd.slice(i) = tmp1.t();
      arma::mat tmp2 = rep_mat(mean_pred.slice(i), counts);
      expanded_mean.slice(i) = tmp2.t();
      intv.slice(i) = intervals(expanded_mean.slice(i), expanded_sd.slice(i),
        probs, n);
    }
    
    
    return Rcpp::List::create(Rcpp::Named("intervals") = intv,
      Rcpp::Named("mean_pred") = expanded_mean,
      Rcpp::Named("sd_pred") = expanded_sd);
  }
}

arma::cube nlg_ssm::predict_sample(const arma::mat& thetasim, 
  const arma::mat& alpha, const arma::uvec& counts, 
  const unsigned int predict_type, const unsigned int nsim) {
  
  unsigned int d = p;
  if (predict_type == 3) d = m;
  arma::mat expanded_theta = rep_mat(thetasim, counts);
  arma::mat expanded_alpha = rep_mat(alpha, counts);
  unsigned int n_samples = expanded_theta.n_cols;
  arma::cube sample(d, n, nsim * n_samples);
  for (unsigned int i = 0; i < n_samples; i++) {
    
    theta = expanded_theta.col(i);
    
    sample.slices(i * nsim, (i + 1) * nsim - 1) = 
      sample_model(expanded_alpha.col(i), predict_type, nsim);
    
  }
  return sample;
}

arma::cube nlg_ssm::sample_model(const arma::vec& a1_sim,
  const unsigned int predict_type, const unsigned int nsim) {
  
  arma::cube alpha(m, n, nsim);
  
  std::normal_distribution<> normal(0.0, 1.0);
  
  // sample states
  for (unsigned int i = 0; i < nsim; i++) {
    
    alpha.slice(i).col(0) = a1_sim;
    for (unsigned int t = 0; t < (n - 1); t++) {
      arma::vec uk(k);
      for(unsigned int j = 0; j < k; j++) {
        uk(j) = normal(engine);
      }
      alpha.slice(i).col(t + 1) = T_fn(t, alpha.slice(i).col(t), 
        theta, known_params, known_tv_params) +  
          R_fn(t, alpha.slice(i).col(t), theta, known_params, known_tv_params) * uk;
    }
  }
  if (predict_type < 3) {
    // construct mean
    arma::cube y_pred(p, n, nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      for (unsigned int t = 0; t < n; t++) {
        y_pred.slice(i).col(t) = Z_fn(t, alpha.slice(i).col(t), theta, 
          known_params, known_tv_params);
      }
    }
    // sample observation noise
    if(predict_type == 1) {
      for (unsigned int i = 0; i < nsim; i++) {
        for (unsigned int t = 0; t < n; t++) {
          arma::vec up(p);
          for (unsigned int j = 0; j < p; j++) {
            up(j) = normal(engine);
          }
          y_pred.slice(i).col(t) += H_fn(t, alpha.slice(i).col(t), 
            theta, known_params, known_tv_params) * up;
        }
      }
    }
    return y_pred;
  }
  return alpha;
  
}

double nlg_ssm::ekf(arma::mat& at, arma::mat& att, arma::cube& Pt, 
  arma::cube& Ptt, const unsigned int iekf_iter) const {
  
  at.col(0) = a1_fn(theta, known_params);
  Pt.slice(0) = P1_fn(theta, known_params);
  
  const double LOG2PI = std::log(2.0 * M_PI);
  double logLik = 0.0;
  
  for (unsigned int t = 0; t < n; t++) {
    
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    
    if (na_y.n_elem < p) {
      
      arma::mat Zg = Z_gn(t, at.col(t), theta, known_params, known_tv_params);
      Zg.rows(na_y).zeros();
      arma::mat HHt = H_fn(t, at.col(t), theta, known_params, known_tv_params);
      HHt = HHt * HHt.t();
      HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
      
      arma::mat Ft = Zg * Pt.slice(t) * Zg.t() + HHt;
      
      // first check to avoid armadillo warnings
        bool chol_ok = Ft.is_finite() && arma::all(Ft.diag() > 0);
      if (!chol_ok) return -std::numeric_limits<double>::infinity();
      arma::mat cholF(p, p);
      chol_ok = arma::chol(cholF, Ft);
      if (!chol_ok) return -std::numeric_limits<double>::infinity();
      
      arma::vec vt = y.col(t) - 
        Z_fn(t, at.col(t), theta, known_params, known_tv_params);
      vt.rows(na_y).zeros();
      
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      arma::mat Kt = Pt.slice(t) * Zg.t() * inv_cholF * inv_cholF.t();
      arma::vec atthat = at.col(t) + Kt * vt;
      double diff = 1.0;
      unsigned int i = 0;
      while (diff > 1e-4 && i < iekf_iter) {
        i++;
        Zg = Z_gn(t, atthat, theta, known_params, known_tv_params);
        Zg.rows(na_y).zeros();
        HHt = H_fn(t, atthat, theta, known_params, known_tv_params);
        HHt = HHt * HHt.t();
        HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
        
        Ft = Zg * Pt.slice(t) * Zg.t() + HHt;
        // first check avoid armadillo warnings
        chol_ok = Ft.is_finite();
        if (!chol_ok) return -std::numeric_limits<double>::infinity();
        chol_ok = arma::chol(cholF, Ft);
        if(!chol_ok) return -std::numeric_limits<double>::infinity();
        
        vt = y.col(t) - 
          Z_fn(t, atthat, theta, known_params, known_tv_params) - 
          Zg * (at.col(t) - atthat);
        vt.rows(na_y).zeros();
        
        inv_cholF = arma::inv(arma::trimatu(cholF));
        Kt = Pt.slice(t) * Zg.t() * inv_cholF * inv_cholF.t();
        
        arma::vec atthat_new = at.col(t) + Kt * vt;
        diff = arma::mean(arma::square(atthat-atthat_new));
        atthat = atthat_new;
      }
      att.col(t) = atthat;
      Ptt.slice(t) = Pt.slice(t) - Kt * Ft * Kt.t();
      
      arma::vec Fv = inv_cholF.t() * vt; 
      logLik -= 0.5 * arma::as_scalar((p - na_y.n_elem) * LOG2PI + 
        2.0 * arma::accu(arma::log(arma::diagvec(cholF))) + Fv.t() * Fv);
    } else {
      att.col(t) = at.col(t);
      Ptt.slice(t) = Pt.slice(t);
    } 
    
    at.col(t + 1) = T_fn(t, att.col(t), theta, known_params, known_tv_params);
    arma::mat Tg = T_gn(t, att.col(t), theta, known_params, known_tv_params);
    arma::mat Rt = R_fn(t, att.col(t), theta, known_params, known_tv_params);
    Pt.slice(t + 1) = Tg * Ptt.slice(t) * Tg.t() + Rt * Rt.t();
  }
  
  return logLik;
}


double nlg_ssm::ekf_loglik(const unsigned int iekf_iter) const {
  
  
  arma::vec at = a1_fn(theta, known_params);
  arma::mat Pt = P1_fn(theta, known_params);
  
  const double LOG2PI = std::log(2.0 * M_PI);
  double logLik = 0.0;
  
  for (unsigned int t = 0; t < n; t++) {
    
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    
    arma::vec att = at;
    arma::mat Ptt = Pt;
    if (na_y.n_elem < p) {
      arma::mat Zg = Z_gn(t, at, theta, known_params, known_tv_params);
      Zg.rows(na_y).zeros();
      arma::mat HHt = H_fn(t, at, theta, known_params, known_tv_params);
      HHt = HHt * HHt.t();
      HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
      
      arma::mat Ft = Zg * Pt * Zg.t() + HHt;
      // first check avoid armadillo warnings
        bool chol_ok = Ft.is_finite() && arma::all(Ft.diag() > 0);
      if (!chol_ok) return -std::numeric_limits<double>::infinity();
      arma::mat cholF(p, p);
      chol_ok = arma::chol(cholF, Ft);
      if (!chol_ok) return -std::numeric_limits<double>::infinity();
      
      arma::vec vt = y.col(t) - 
        Z_fn(t, at, theta, known_params, known_tv_params);
      vt.rows(na_y).zeros();
      
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      arma::mat Kt = Pt * Zg.t() * inv_cholF * inv_cholF.t();
      
      arma::vec atthat = at + Kt * vt;
      double diff = 1.0;
      unsigned int i = 0;
      while (diff > 1e-4 && i < iekf_iter) {
        i++;
        Zg = Z_gn(t, atthat, theta, known_params, known_tv_params);
        Zg.rows(na_y).zeros();
        HHt = H_fn(t, atthat, theta, known_params, known_tv_params);
        HHt = HHt * HHt.t();
        HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
        
        Ft = Zg * Pt * Zg.t() + HHt;
        
        // first check avoid armadillo warnings
        chol_ok = Ft.is_finite();
        if (!chol_ok) return -std::numeric_limits<double>::infinity();
        chol_ok = arma::chol(cholF, Ft);
        if (!chol_ok) return -std::numeric_limits<double>::infinity();
        
        vt = y.col(t) - 
          Z_fn(t, atthat, theta, known_params, known_tv_params) - 
          Zg * (at.col(t) - atthat);
        vt.rows(na_y).zeros();
        
        inv_cholF = arma::inv(arma::trimatu(cholF));
        Kt = Pt * Zg.t() * inv_cholF * inv_cholF.t();
        
        arma::vec atthat_new = at + Kt * vt;
        diff = arma::mean(arma::square(atthat-atthat_new));
        atthat = atthat_new;
      }
      att = atthat;
      
      Ptt -= Kt * Ft * Kt.t();
      
      arma::vec Fv = inv_cholF.t() * vt; 
      logLik -= 0.5 * arma::as_scalar((p - na_y.n_elem) * LOG2PI + 
        2.0 * arma::accu(arma::log(arma::diagvec(cholF))) + Fv.t() * Fv);
    }
    
    at = T_fn(t, att, theta, known_params, known_tv_params);
    
    arma::mat Tg = T_gn(t, att, theta, known_params, known_tv_params);
    arma::mat Rt = R_fn(t, att, theta, known_params, known_tv_params);
    Pt = Tg * Ptt * Tg.t() + Rt * Rt.t();
    
  }
  return logLik;
}

double nlg_ssm::ekf_smoother(arma::mat& at, arma::cube& Pt, const unsigned int iekf_iter) const {
  
  at.col(0) = a1_fn(theta, known_params);
  
  Pt.slice(0) = P1_fn(theta, known_params);
  
  arma::mat vt(p, n,arma::fill::zeros);
  arma::cube ZFinv(m, p, n,arma::fill::zeros);
  arma::cube Kt(m, p, n,arma::fill::zeros);
  
  arma::uvec obs(n, arma::fill::ones);
  
  arma::mat att(m, n);
  
  const double LOG2PI = std::log(2.0 * M_PI);
  double logLik = 0.0;
  
  for (unsigned int t = 0; t < n; t++) {
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    arma::mat Ptt = Pt.slice(t);
    
    if (na_y.n_elem < p) {
      
      arma::mat Zg = Z_gn(t, at.col(t), theta, known_params, known_tv_params);
      Zg.rows(na_y).zeros();
      arma::mat HHt = H_fn(t, at.col(t), theta, known_params, known_tv_params);
      HHt = HHt * HHt.t();
      HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
      arma::mat Ft = Zg * Pt.slice(t) * Zg.t() + HHt;
      // first check avoid armadillo warnings
        bool chol_ok = Ft.is_finite() && arma::all(Ft.diag() > 0);
      if (!chol_ok) return -std::numeric_limits<double>::infinity();
      arma::mat cholF(p, p);
      chol_ok = arma::chol(cholF, Ft);
      if (!chol_ok) return -std::numeric_limits<double>::infinity();
      
      vt.col(t) = y.col(t) - 
        Z_fn(t, at.col(t), theta, known_params, known_tv_params);
      vt.rows(na_y).zeros();
      
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      ZFinv.slice(t) = Zg.t() * inv_cholF * inv_cholF.t();
      Kt.slice(t) = Pt.slice(t) * ZFinv.slice(t);
      arma::vec atthat = at.col(t) + Kt.slice(t) * vt.col(t);
      
      double diff = 1.0;
      unsigned int i = 0;
      while (diff > 1e-4 && i < iekf_iter) {
        i++;
        
        Zg = Z_gn(t, atthat, theta, known_params, known_tv_params);
        Zg.rows(na_y).zeros();
        HHt = H_fn(t, atthat, theta, known_params, known_tv_params);
        HHt = HHt * HHt.t();
        HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
        
        Ft = Zg * Pt.slice(t) * Zg.t() + HHt;
        // first check avoid armadillo warnings
        chol_ok = Ft.is_finite();
        if (!chol_ok) return -std::numeric_limits<double>::infinity();
        chol_ok = arma::chol(cholF, Ft);
        if (!chol_ok) return -std::numeric_limits<double>::infinity();
        
        vt.col(t) = y.col(t) - 
          Z_fn(t, atthat, theta, known_params, known_tv_params) - 
          Zg * (at.col(t) - atthat);
        vt.rows(na_y).zeros();
        
        inv_cholF = arma::inv(arma::trimatu(cholF));
        Kt.slice(t) = Pt.slice(t) * Zg.t() * inv_cholF * inv_cholF.t();
        
        arma::vec atthat_new = at.col(t) + Kt.slice(t) * vt.col(t);
        
        diff = arma::mean(arma::square(atthat-atthat_new));
        atthat = atthat_new;
      }
      att.col(t) = atthat;
      Ptt = Pt.slice(t) - Kt.slice(t) * Ft * Kt.slice(t).t();
      arma::vec Fv = inv_cholF.t() * vt.col(t); 
      logLik -= 0.5 * arma::as_scalar((p - na_y.n_elem) * LOG2PI + 
        2.0 * arma::accu(arma::log(arma::diagvec(cholF))) + Fv.t() * Fv);
    } else {
      att.col(t) = at.col(t);
      obs(t) = 0;
    }
    
    at.col(t + 1) = T_fn(t, att.col(t), theta, known_params, known_tv_params);
    arma::mat Tg = T_gn(t, att.col(t), theta, known_params, known_tv_params);
    arma::mat Rt = R_fn(t, att.col(t), theta, known_params, known_tv_params);
    Pt.slice(t + 1) = Tg * Ptt * Tg.t() + Rt * Rt.t();
  }
  
  
  arma::vec rt(m, arma::fill::zeros);
  arma::mat Nt(m, m, arma::fill::zeros);
  for (int t = (n - 1); t >= 0; t--) {
    arma::mat Tg = T_gn(t, att.col(t), theta, known_params, known_tv_params);
    if (obs(t)) {
      arma::mat Zg = Z_gn(t, at.col(t), theta, known_params, known_tv_params);
      arma::mat L = Tg * (arma::eye(m, m) - Kt.slice(t) * Zg);
      rt = ZFinv.slice(t) * vt.col(t) + L.t() * rt;
      Nt = arma::symmatu(ZFinv.slice(t) * Zg + L.t() * Nt * L);
    } else {
      rt = Tg.t() * rt;
      Nt = Tg.t() * Nt * Tg;
    }
    at.col(t) += Pt.slice(t) * rt;
    Pt.slice(t) -= Pt.slice(t) * Nt * Pt.slice(t);
  }
  return logLik;
}

double nlg_ssm::ekf_fast_smoother(arma::mat& at, const unsigned int iekf_iter) const {
  
  at.col(0) = a1_fn(theta, known_params);
  
  arma::cube Pt(m, m, n + 1);
  
  Pt.slice(0) = P1_fn(theta, known_params);
  
  arma::mat vt(p, n,arma::fill::zeros);
  arma::cube ZFinv(m, p, n,arma::fill::zeros);
  arma::cube Kt(m, p, n,arma::fill::zeros);
  
  arma::uvec obs(n, arma::fill::ones);
  
  arma::mat att(m, n);
  
  const double LOG2PI = std::log(2.0 * M_PI);
  double logLik = 0.0;
  
  for (unsigned int t = 0; t < n; t++) {
    
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    arma::mat Ptt = Pt.slice(t);
    
    if (na_y.n_elem < p) {
      
      arma::mat Zg = Z_gn(t, at.col(t), theta, known_params, known_tv_params);
      Zg.rows(na_y).zeros();
      arma::mat HHt = H_fn(t, at.col(t), theta, known_params, known_tv_params);
      HHt = HHt * HHt.t();
      HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
      
      arma::mat Ft = Zg * Pt.slice(t) * Zg.t() + HHt;
      // first check avoid armadillo warnings
        bool chol_ok = Ft.is_finite() && arma::all(Ft.diag() > 0);
      if (!chol_ok) return -std::numeric_limits<double>::infinity();
      arma::mat cholF(p, p);
      chol_ok = arma::chol(cholF, Ft);
      if (!chol_ok) return -std::numeric_limits<double>::infinity();
      
      vt.col(t) = y.col(t) - 
        Z_fn(t, at.col(t), theta, known_params, known_tv_params);
      vt.rows(na_y).zeros();
      
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      ZFinv.slice(t) = Zg.t() * inv_cholF * inv_cholF.t();
      Kt.slice(t) = Pt.slice(t) * ZFinv.slice(t);
      
      
      arma::vec atthat = at.col(t) + Kt.slice(t) * vt.col(t);
      double diff = 1.0;
      unsigned int i = 0;
      while (diff > 1e-4 && i < iekf_iter) {
        i++;
        Zg = Z_gn(t, atthat, theta, known_params, known_tv_params);
        Zg.rows(na_y).zeros();
        HHt =H_fn(t, atthat, theta, known_params, known_tv_params);
        HHt = HHt * HHt.t();
        HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
        
        Ft = Zg * Pt.slice(t) * Zg.t() + HHt;
        
        // first check avoid armadillo warnings
        chol_ok = Ft.is_finite();
        if (!chol_ok) return -std::numeric_limits<double>::infinity();
        chol_ok = arma::chol(cholF, Ft);
        if (!chol_ok) return -std::numeric_limits<double>::infinity();
        
        vt.col(t) = y.col(t) - 
          Z_fn(t, atthat, theta, known_params, known_tv_params) - 
          Zg * (at.col(t) - atthat);
        vt.rows(na_y).zeros();
        
        inv_cholF = arma::inv(arma::trimatu(cholF));
        Kt.slice(t) = Pt.slice(t) * Zg.t() * inv_cholF * inv_cholF.t();
        
        arma::vec atthat_new = at.col(t) + Kt.slice(t) * vt.col(t);
        diff = arma::mean(arma::square(atthat-atthat_new));
        atthat = atthat_new;
      }
      att.col(t) = atthat;
      Ptt = Pt.slice(t) - Kt.slice(t) * Ft * Kt.slice(t).t();
      
      arma::vec Fv = inv_cholF.t() * vt.col(t); 
      logLik -= 0.5 * arma::as_scalar((p - na_y.n_elem) * LOG2PI + 
        2.0 * arma::accu(arma::log(arma::diagvec(cholF))) + Fv.t() * Fv);
    } else {
      att.col(t) = at.col(t);
      obs(t) = 0;
    }
    
    at.col(t + 1) = T_fn(t, att.col(t), theta, known_params, known_tv_params);
    arma::mat Tg = T_gn(t, att.col(t), theta, known_params, known_tv_params);
    arma::mat Rt = R_fn(t, att.col(t), theta, known_params, known_tv_params);
    Pt.slice(t + 1) = Tg * Ptt * Tg.t() + Rt * Rt.t();
    
  }
  
  
  arma::vec rt(m, arma::fill::zeros);
  for (int t = (n - 1); t >= 0; t--) {
    arma::mat Tg = T_gn(t, att.col(t), theta, known_params, known_tv_params);
    if (obs(t)) {
      arma::mat Zg = Z_gn(t, at.col(t), theta, known_params, known_tv_params);
      arma::mat L = Tg * (arma::eye(m, m) - Kt.slice(t) * Zg);
      rt = ZFinv.slice(t) * vt.col(t) + L.t() * rt;
    } else {
      rt = Tg.t() * rt;
    }
    at.col(t) += Pt.slice(t) * rt;
  }
  
  return logLik;
}

// Unscented Kalman filter, Särkkä (2013) p.107 (UKF) and
// Note that the initial distribution is given for alpha_1
// so we first do update instead of prediction
double nlg_ssm::ukf(arma::mat& at, arma::mat& att, arma::cube& Pt, 
  arma::cube& Ptt, const double alpha, const double beta, const double kappa) const {
  
  // // Parameters of UKF, currently fixed for simplicity
  // These are from Särkkä?
  // double alpha = 1.0;
  // double beta = 0.0;
  // double kappa = 4.0;
  // van der Merwe et al:
  // alpha something small...
  // beta = 4? 
  // kappa = 0
  //DK? alpha=1, beta=0, kappa 3-m?
  const double LOG2PI = std::log(2.0 * M_PI);
  double logLik = 0.0;
  
  double lambda = alpha * alpha * (m + kappa) - m;
  
  unsigned int n_sigma = 2 * m + 1;
  arma::vec wm(n_sigma);
  wm(0) = lambda / (lambda + m);
  wm.subvec(1, n_sigma - 1).fill(1.0 / (2.0 * (lambda + m)));
  arma::vec wc = wm;
  wc(0) +=  1.0 - alpha * alpha + beta;
  
  
  double sqrt_m_lambda = std::sqrt(m + lambda);
  
  at.col(0) = a1_fn(theta, known_params);
  Pt.slice(0) = P1_fn(theta, known_params);
  
  for (unsigned int t = 0; t < n; t++) {
    // update step
    
    // compute cholesky of Pt
    arma::mat cholP(m, m);
    cholP = psd_chol(Pt.slice(t));
    
    // form the sigma points
    arma::mat sigma(m, n_sigma);
    sigma.col(0) = at.col(t);
    for (unsigned int i = 1; i <= m; i++) {
      sigma.col(i) = at.col(t) + sqrt_m_lambda * cholP.col(i - 1);
      sigma.col(i + m) = at.col(t) - sqrt_m_lambda * cholP.col(i - 1);
    }
    
    arma::uvec obs_y = arma::find_finite(y.col(t));
    
    if (obs_y.n_elem > 0) {
      
      // propagate sigma points
      arma::mat sigma_y(obs_y.n_elem, n_sigma);
      for (unsigned int i = 0; i < n_sigma; i++) {
        sigma_y.col(i) = Z_fn(t, sigma.col(i), theta, known_params, known_tv_params).rows(obs_y);
      }
      arma::vec pred_mean = sigma_y * wm;
      arma::mat pred_var = H_fn(t, at.col(t), theta, known_params, known_tv_params).submat(obs_y, obs_y);
      arma::mat pred_cov(m, obs_y.n_elem, arma::fill::zeros);
      for (unsigned int i = 0; i < n_sigma; i++) {
        arma::vec tmp = sigma_y.col(i) - pred_mean;
        pred_var += wc(i) * tmp * tmp.t();
        pred_cov += wc(i) * (sigma.col(i) - at.col(t)) * tmp.t();
      }
      // filtered estimates
      arma::vec v = arma::mat(y.rows(obs_y)).col(t) - pred_mean;
      arma::mat K = arma::solve(pred_var, pred_cov.t()).t();
      att.col(t) = at.col(t) + K * v;
      Ptt.slice(t) = Pt.slice(t) - K * pred_var * K.t();
      
      arma::mat cholF = arma::chol(pred_var);
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      arma::vec Fv = inv_cholF * v; 
      logLik -= 0.5 * arma::as_scalar(obs_y.n_elem * LOG2PI + 
        2.0 * arma::accu(arma::log(arma::diagvec(cholF))) + Fv.t() * Fv);
    } else {
      att.col(t) = at.col(t);
      Ptt.slice(t) = Pt.slice(t);
    }
    
    // prediction
    // compute cholesky of Ptt
    arma::mat cholPtt = psd_chol(Ptt.slice(t));
    
    // form the sigma points and propagate
    sigma.col(0) = T_fn(t, att.col(t), theta, known_params, known_tv_params);
    for (unsigned int i = 1; i <= m; i++) {
      sigma.col(i) = T_fn(t, att.col(t) + sqrt_m_lambda * cholPtt.col(i - 1), 
        theta, known_params, known_tv_params);
      sigma.col(i + m) = T_fn(t, att.col(t) - sqrt_m_lambda * cholPtt.col(i - 1), 
        theta, known_params, known_tv_params);
    }
    
    at.col(t + 1) = sigma * wm;
    
    arma::mat Rt = R_fn(t, att.col(t), theta, known_params, known_tv_params);
    Pt.slice(t + 1) = Rt * Rt.t();
    for (unsigned int i = 0; i < n_sigma; i++) {
      arma::vec tmp = sigma.col(i) - at.col(t + 1);
      Pt.slice(t + 1) += wc(i) * tmp * tmp.t();
    }
  }
  return logLik;
}

mgg_ssm nlg_ssm::approximate(arma::mat& mode_estimate, 
  const unsigned int max_iter, const double conv_tol, 
  const unsigned int iekf_iter) const {
  
  // initial approximation is based on EKF (at and att)
  arma::mat at(m, n + 1);
  arma::mat att(m, n);
  arma::cube Pt(m, m, n + 1);
  arma::cube Ptt(m, m, n);
  double loglik = ekf(at, att, Pt, Ptt, iekf_iter);
  if(!arma::is_finite(loglik)) {
    mode_estimate.fill(std::numeric_limits<double>::infinity());
    arma::vec a1 = a1_fn(theta, known_params);
    arma::mat P1 = P1_fn(theta, known_params);
    arma::cube Z(p, m, n);
    arma::cube H(p, p, (n - 1) * Htv + 1);
    arma::cube T(m, m, n);
    arma::cube R(m, k, (n - 1) * Rtv + 1);
    arma::mat D(p, n,arma::fill::zeros);
    arma::mat C(m, n,arma::fill::zeros);
    mgg_ssm approx_model(y, Z, H, T, R, a1, P1, arma::cube(0,0,0),
      arma::mat(0,0), D, C, seed);
    return approx_model;
  }
  arma::vec a1 = a1_fn(theta, known_params);
  arma::mat P1 = P1_fn(theta, known_params);
  arma::cube Z(p, m, n);
  for (unsigned int t = 0; t < Z.n_slices; t++) {
    Z.slice(t) = Z_gn(t, at.col(t), theta, known_params, known_tv_params);
  }
  arma::cube H(p, p, (n - 1) * Htv + 1);
  for (unsigned int t = 0; t < H.n_slices; t++) {
    H.slice(t) = H_fn(t, at.col(t), theta, known_params, known_tv_params);
  }
  arma::cube T(m, m, n);
  for (unsigned int t = 0; t < T.n_slices; t++) {
    T.slice(t) = T_gn(t, att.col(t), theta, known_params, known_tv_params);
  }
  
  arma::cube R(m, k, (n - 1) * Rtv + 1);
  for (unsigned int t = 0; t < R.n_slices; t++) {
    R.slice(t) = R_fn(t, att.col(t), theta, known_params, known_tv_params);
  }
  arma::mat D(p, n,arma::fill::zeros);
  arma::mat C(m, n,arma::fill::zeros);
  
  for (unsigned int t = 0; t < n; t++) {
    D.col(t) = Z_fn(t, at.col(t), theta, known_params, known_tv_params) -
      Z.slice(t) * at.col(t);
    C.col(t) =  T_fn(t, att.col(t), theta, known_params, known_tv_params) -
      T.slice(t) * att.col(t);
  }
  
  mgg_ssm approx_model(y, Z, H, T, R, a1, P1, arma::cube(0,0,0),
    arma::mat(0,0), D, C, seed);
  // Refine approximation iteratively
  mode_estimate = approximate(approx_model, max_iter, conv_tol);
  
  return approx_model;
}

arma::mat nlg_ssm::approximate(mgg_ssm& approx_model,
  const unsigned int max_iter, const double conv_tol) const {
  

  //check model
  arma::mat mode_estimate = approx_model.fast_smoother().head_cols(n);
  if (!arma::is_finite(mode_estimate)) {
    return mode_estimate;
  }
  double ll;
  if (max_iter > 0) ll = log_signal_pdf(mode_estimate);
  unsigned int i = 0;
  double rel_diff = 1.0e300; 
  double abs_diff = 1;
  
  while(i < max_iter && rel_diff > conv_tol && abs_diff > 1e-4) {
    
    i++;
    for (unsigned int t = 0; t < approx_model.Z.n_slices; t++) {
      approx_model.Z.slice(t) = Z_gn(t, mode_estimate.col(t), theta, known_params, known_tv_params);
    }
    
    for (unsigned int t = 0; t < approx_model.T.n_slices; t++) {
      approx_model.T.slice(t) = T_gn(t, mode_estimate.col(t), theta, known_params, known_tv_params);
    }
    
    for (unsigned int t = 0; t < n; t++) {
      approx_model.D.col(t) = Z_fn(t, mode_estimate.col(t), theta, known_params, known_tv_params) -
        approx_model.Z.slice(t * Zgtv) * mode_estimate.col(t);
      approx_model.C.col(t) =  T_fn(t, mode_estimate.col(t), theta, known_params, known_tv_params) -
        approx_model.T.slice(t * Tgtv) * mode_estimate.col(t);
    }
    for (unsigned int t = 0; t < approx_model.H.n_slices; t++) {
      approx_model.H.slice(t) = H_fn(t, mode_estimate.col(t), theta, known_params, known_tv_params);
    }
    for (unsigned int t = 0; t < approx_model.R.n_slices; t++) {
      approx_model.R.slice(t) = R_fn(t, mode_estimate.col(t), theta, known_params, known_tv_params);
    }
    approx_model.compute_HH();
    approx_model.compute_RR();
    
    // compute new value of mode
    arma::mat mode_estimate_new = approx_model.fast_smoother().head_cols(n);
    double ll_new = log_signal_pdf(mode_estimate_new);
    abs_diff = ll_new - ll;
    rel_diff = abs_diff / std::abs(ll);
    if (!arma::is_finite(mode_estimate_new) || !arma::is_finite(ll_new)) {
      mode_estimate.fill(std::numeric_limits<double>::infinity());
      return mode_estimate;
    }
    if(rel_diff < -conv_tol && i > 1 && abs_diff > 1e-4) {
      
      unsigned int ii = 0;
      double step_size = 1.0;
      // we went too far with previous mode_estimate
      // backtrack between mode_estimate_old and mode_estimate
      arma::mat mode_estimate_old = mode_estimate;
      while(rel_diff < -conv_tol && ii < 15 && abs_diff > 1e-4) {
        step_size = step_size / 2.0;
        mode_estimate = (1.0 - step_size) * mode_estimate_old + step_size * mode_estimate_new;
        
        ll_new = log_signal_pdf(mode_estimate);
        abs_diff = ll_new - ll;
        rel_diff = abs_diff / std::abs(ll);
        ii++;
        if (!arma::is_finite(mode_estimate) || !arma::is_finite(ll_new)) {
          mode_estimate.fill(std::numeric_limits<double>::infinity());
          return mode_estimate;
        }
      }
      if (ii == 15) {
        mode_estimate.fill(std::numeric_limits<double>::infinity());
        return mode_estimate;
      }
      mode_estimate_new = mode_estimate;
    }
    mode_estimate = mode_estimate_new;
    ll = ll_new;
    
  }
  if (i == max_iter && max_iter > 0) {
    mode_estimate.fill(std::numeric_limits<double>::infinity());
    return mode_estimate;
  }
  
  return mode_estimate;
}

arma::vec nlg_ssm::log_weights(const mgg_ssm& approx_model, 
  const unsigned int t, const arma::cube& alpha, const arma::mat& alpha_prev) const {
  
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  
  if (arma::is_finite(y(t))) {
    
    // original H depends on time or state <=> approx H depends on time or state
    if(Htv == 1) {
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        weights(i) = 
          dmvnorm(y.col(t), Z_fn(t, alpha.slice(i).col(t), theta, known_params, known_tv_params), 
            H_fn(t, alpha.slice(i).col(t), theta, known_params, known_tv_params), true, true) -
              dmvnorm(y.col(t), approx_model.D.col(t) + approx_model.Z.slice(t * approx_model.Ztv) * alpha.slice(i).col(t),  
                approx_model.H.slice(t * approx_model.Htv), true, true);
      }
    } else {
      arma::mat H = H_fn(t, alpha.slice(0).col(t), theta, known_params, known_tv_params);
      arma::uvec nonzero = arma::find(H.diag() > (std::numeric_limits<double>::epsilon() * H.n_cols * H.diag().max()));
      arma::mat Linv(nonzero.n_elem, nonzero.n_elem);
      double constant = precompute_dmvnorm(H, Linv, nonzero);
      
      arma::mat H_a = approx_model.H.slice(0);
      arma::uvec nonzero_a = arma::find(H_a.diag() > (std::numeric_limits<double>::epsilon() * H_a.n_cols * H_a.diag().max()));
      arma::mat Linv_a(nonzero_a.n_elem, nonzero_a.n_elem);
      double constant_a = precompute_dmvnorm(H_a, Linv_a, nonzero_a);
      
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        weights(i) = fast_dmvnorm(y.col(t), Z_fn(t, alpha.slice(i).col(t), 
          theta, known_params, known_tv_params), Linv, nonzero, constant) -
            fast_dmvnorm(y.col(t), approx_model.D.col(t) + 
            approx_model.Z.slice(t * approx_model.Ztv) * alpha.slice(i).col(t),  
            Linv_a, nonzero_a, constant_a);
      }
    }
  }
  arma::vec weights_t(alpha.n_slices, arma::fill::zeros);
  if(t > 0) {
    
    for (unsigned int i = 0; i < alpha.n_slices; i++) {
      
      arma::vec mean = T_fn(t - 1, alpha_prev.col(i), theta, known_params, known_tv_params);
      arma::mat cov = R_fn(t - 1, alpha_prev.col(i), theta, known_params, known_tv_params);
      cov = cov * cov.t();
      arma::vec approx_mean = approx_model.C.col(t - 1) + 
        approx_model.T.slice((t - 1) * approx_model.Ttv) * alpha_prev.col(i);
      
      weights_t(i) +=  dmvnorm(alpha.slice(i).col(t), approx_mean, 
        approx_model.RR.slice((t - 1) * approx_model.Rtv), false, true) -
          dmvnorm(alpha.slice(i).col(t), mean, cov, false, true);
      weights_t(i) = log1pexp(weights_t(i));
    }
  }
  
  return weights - weights_t;
}

// compute _normalized_ mode-based scaling terms
// log[g(y_t | ^alpha_t) f(^alpha_t | ^alpha_t-1) / 
// ~g(y_t | ^alpha_t)] ~f(^alpha_t | ^alpha_t-1)
arma::vec nlg_ssm::scaling_factors(const mgg_ssm& approx_model,
  const arma::mat& mode_estimate) const {
  
  arma::vec weights(n, arma::fill::zeros);
  
  for(unsigned int t = 0; t < n; t++) {
    if (arma::is_finite(y(t))) {
      weights(t) =  dmvnorm(y.col(t), Z_fn(t, mode_estimate.col(t), theta, known_params, known_tv_params),
        H_fn(t, mode_estimate.col(t), theta, known_params, known_tv_params), true, true) -
          dmvnorm(y.col(t), approx_model.D.col(t) + approx_model.Z.slice(t * approx_model.Ztv) * mode_estimate.col(t),
            approx_model.H.slice(t * approx_model.Htv), true, true);
    }
  }
  
  for (unsigned int t = 1; t < n; t++) {
    arma::vec mean = T_fn(t-1, mode_estimate.col(t-1), theta, known_params, known_tv_params);
    arma::mat cov = R_fn(t-1, mode_estimate.col(t-1), theta, known_params, known_tv_params);
    cov = cov * cov.t();
    arma::vec approx_mean = approx_model.C.col(t-1) +
      approx_model.T.slice((t-1) * approx_model.Ttv) * mode_estimate.col(t-1);
    
    weights(t) += dmvnorm(mode_estimate.col(t), mean, cov, false, true) -
      dmvnorm(mode_estimate.col(t), approx_mean, approx_model.RR.slice((t-1) * approx_model.Rtv), false, true);
    
  }
  
  return weights;
}

// Logarithms of _normalized_ densities g(y_t | alpha_t)
/*
 * t:             Time point where the densities are computed
 * alpha:         Simulated particles
 */
arma::vec nlg_ssm::log_obs_density(const unsigned int t, 
  const arma::cube& alpha) const {
  
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  
  if (arma::is_finite(y(t))) {
    for (unsigned int i = 0; i < alpha.n_slices; i++) {
      weights(i) = dmvnorm(y.col(t), Z_fn(t, alpha.slice(i).col(t), theta, known_params, known_tv_params), 
        H_fn(t, alpha.slice(i).col(t), theta, known_params, known_tv_params), true, true);
    }
  }
  return weights;
}

double nlg_ssm::log_obs_density(const unsigned int t, 
  const arma::vec& alpha) const {
  
  double weight = 0.0;
  
  if (arma::is_finite(y(t))) {
    weight = dmvnorm(y.col(t), Z_fn(t, alpha, theta, known_params, known_tv_params), 
      H_fn(t, alpha, theta, known_params, known_tv_params), true, true);
    
  }
  return weight;
}

// apart from using mgg_ssm, identical with ung_ssm::psi_filter
double nlg_ssm::psi_filter(const mgg_ssm& approx_model,
  const double approx_loglik,
  const unsigned int nsim, arma::cube& alpha, arma::mat& weights,
  arma::umat& indices) {
  
  arma::mat alphahat(m, n + 1);
  arma::cube Vt(m, m, n + 1);
  arma::cube Ct(m, m, n + 1);
  approx_model.smoother_ccov(alphahat, Vt, Ct);
  if (!Vt.is_finite() || !Ct.is_finite()) {
    return -std::numeric_limits<double>::infinity();
  }
  conditional_cov(Vt, Ct);
  std::normal_distribution<> normal(0.0, 1.0);
  
  for (unsigned int i = 0; i < nsim; i++) {
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    alpha.slice(i).col(0) = alphahat.col(0) + Vt.slice(0) * um;
  }
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec normalized_weights(nsim);
  double loglik = 0.0;
  if(arma::is_finite(y(0))) {
    //weights.col(0) = exp(log_weights(approx_model, 0, alpha) - scales(0));
    weights.col(0) = log_weights(approx_model, 0, alpha, arma::mat(m, nsim, arma::fill::zeros));
    double max_weight = weights.col(0).max();
    weights.col(0) = arma::exp(weights.col(0) - max_weight);
    double sum_weights = arma::accu(weights.col(0));
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -std::numeric_limits<double>::infinity();
    }
    loglik = max_weight + approx_loglik + std::log(sum_weights / nsim);
  } else {
    weights.col(0).ones();
    normalized_weights.fill(1.0 / nsim);
    loglik = approx_loglik;
  }
  
  for (unsigned int t = 0; t < n; t++) {
    arma::vec r(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    indices.col(t) = stratified_sample(normalized_weights, r, nsim);
    
    arma::mat alphatmp(m, nsim);
    
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp.col(i) = alpha.slice(indices(i, t)).col(t);
    }
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec um(m);
      for(unsigned int j = 0; j < m; j++) {
        um(j) = normal(engine);
      }
      alpha.slice(i).col(t + 1) = alphahat.col(t + 1) +
        Ct.slice(t + 1) * (alphatmp.col(i) - alphahat.col(t)) + Vt.slice(t + 1) * um;
    }
    
    if ((t < (n - 1)) && arma::is_finite(y(t + 1))) {
      
      weights.col(t + 1) = log_weights(approx_model, t + 1, alpha, alphatmp);
      double max_weight = weights.col(t+1).max();
      weights.col(t+1) = arma::exp(weights.col(t+1) - max_weight);
      double sum_weights = arma::accu(weights.col(t + 1));
      if(sum_weights > 0.0){
        normalized_weights = weights.col(t + 1) / sum_weights;
      } else {
        return -std::numeric_limits<double>::infinity();
      }
      loglik += max_weight + std::log(sum_weights / nsim);
    } else {
      weights.col(t + 1).ones();
      normalized_weights.fill(1.0 / nsim);
    }
  }
  
  return loglik;
}


// Logarithms of _normalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
/*
 * approx_model:  Gaussian approximation of the original model
 * t:             Time point where the weights are computed
 * alpha:         Simulated particles
 */

double nlg_ssm::bsf_filter(const unsigned int nsim, arma::cube& alpha,
  arma::mat& weights, arma::umat& indices) {
  
  arma::vec a1 = a1_fn(theta, known_params);
  arma::mat P1 = P1_fn(theta, known_params);
  arma::uvec nonzero = arma::find(P1.diag() > 0);
  arma::mat L_P1 = psd_chol(P1);
  std::normal_distribution<> normal(0.0, 1.0);
  for (unsigned int i = 0; i < nsim; i++) {
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    alpha.slice(i).col(0) = a1 + L_P1 * um;
    
  }
  
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec normalized_weights(nsim);
  double loglik = 0.0;
  
  if(arma::is_finite(y(0))) {
    weights.col(0) = log_obs_density(0, alpha);
    double max_weight = weights.col(0).max();
    weights.col(0) = arma::exp(weights.col(0) - max_weight);
    double sum_weights = arma::accu(weights.col(0));
    
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -std::numeric_limits<double>::infinity();
    }
    loglik = max_weight + std::log(sum_weights / nsim);
  } else {
    weights.col(0).ones();
    normalized_weights.fill(1.0 / nsim);
  }
  for (unsigned int t = 0; t < n; t++) {
    
    arma::vec r(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    
    indices.col(t) = stratified_sample(normalized_weights, r, nsim);
    
    arma::mat alphatmp(m, nsim);
    
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp.col(i) = alpha.slice(indices(i, t)).col(t);
    }
    
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec uk(k);
      for(unsigned int j = 0; j < k; j++) {
        uk(j) = normal(engine);
      }
      alpha.slice(i).col(t + 1) = T_fn(t, alphatmp.col(i), theta, known_params, known_tv_params) + 
        R_fn(t, alphatmp.col(i), theta, known_params, known_tv_params) * uk;
    }
    
    if ((t < (n - 1)) && arma::is_finite(y(t + 1))) {
      weights.col(t + 1) = log_obs_density(t + 1, alpha);
      
      double max_weight = weights.col(t + 1).max();
      weights.col(t + 1) = arma::exp(weights.col(t + 1) - max_weight);
      double sum_weights = arma::accu(weights.col(t + 1));
      if(sum_weights > 0.0){
        normalized_weights = weights.col(t + 1) / sum_weights;
      } else {
        return -std::numeric_limits<double>::infinity();
      }
      loglik += max_weight + std::log(sum_weights / nsim);
    } else {
      weights.col(t + 1).ones();
      normalized_weights.fill(1.0/nsim);
    }
  }
  return loglik;
}


double nlg_ssm::aux_filter(const unsigned int nsim, arma::cube& alpha,
  arma::mat& weights, arma::umat& indices) {
  
  arma::vec a1 = a1_fn(theta, known_params);
  arma::mat P1 = P1_fn(theta, known_params);
  arma::mat L_P1 = psd_chol(P1);
  std::normal_distribution<> normal(0.0, 1.0);
  for (unsigned int i = 0; i < nsim; i++) {
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    alpha.slice(i).col(0) = a1 + L_P1 * um;
    
  }
  
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec normalized_weights(nsim);
  double loglik = 0.0;
  
  if(arma::is_finite(y(0))) {
    weights.col(0) = log_obs_density(0, alpha);
    double max_weight = weights.col(0).max();
    weights.col(0) = arma::exp(weights.col(0) - max_weight);
    double sum_weights = arma::accu(weights.col(0));
    
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -std::numeric_limits<double>::infinity();
    }
    loglik = max_weight + std::log(sum_weights / nsim);
  } else {
    weights.col(0).ones();
    normalized_weights.fill(1.0 / nsim);
  }
  
  for (unsigned int t = 0; t < n; t++) {
    
    arma::vec r(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    
    arma::uvec indices_init = stratified_sample(normalized_weights, r, nsim);
    
    arma::mat alphatmp_init(m, nsim);
    arma::vec aux_weights(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp_init.col(i) = alpha.slice(indices_init(i)).col(t);
      aux_weights(i) = log_obs_density(t+1, 
        T_fn(t, alphatmp_init.col(i), theta, known_params, known_tv_params));
    }
    
    double max_aux_weight = aux_weights.max();
    arma::vec normalized_aux_weights = arma::exp(aux_weights-max_aux_weight);
    double sum_aux_weights = arma::accu(normalized_aux_weights);
    normalized_aux_weights /= sum_aux_weights;
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    indices.col(t) = stratified_sample(normalized_aux_weights, r, nsim);
    
    arma::mat alphatmp(m, nsim);
    arma::vec sampled_weights(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp.col(i) = alphatmp_init.col(indices(i, t));
      sampled_weights(i) = aux_weights(indices(i, t));
      indices(i, t) = indices_init(indices(i, t));
    }
    
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec uk(k);
      for(unsigned int j = 0; j < k; j++) {
        uk(j) = normal(engine);
      }
      alpha.slice(i).col(t + 1) = T_fn(t, alphatmp.col(i), theta, known_params, known_tv_params) + 
        R_fn(t, alphatmp.col(i), theta, known_params, known_tv_params) * uk;
    }
    
    if ((t < (n - 1)) && arma::is_finite(y(t + 1))) {
      weights.col(t + 1) = log_obs_density(t + 1, alpha) - sampled_weights;
      
      double max_weight = weights.col(t + 1).max();
      weights.col(t + 1) = arma::exp(weights.col(t + 1) - max_weight);
      double sum_weights = arma::accu(weights.col(t + 1));
      if(sum_weights > 0.0){
        normalized_weights = weights.col(t + 1) / sum_weights;
      } else {
        return -std::numeric_limits<double>::infinity();
      }
      loglik += max_weight + std::log(sum_weights / nsim) + std::log(sum_aux_weights / nsim) +
        max_aux_weight;
    } else {
      weights.col(t + 1).ones();
      normalized_weights.fill(1.0/nsim);
    }
  }
  return loglik;
}


// EKF-based particle filter (van der Merwe et al)

double nlg_ssm::ekf_filter(const unsigned int nsim, arma::cube& alpha,
  arma::mat& weights, arma::umat& indices) {
  
  arma::vec a1 = a1_fn(theta, known_params);
  arma::mat P1 = P1_fn(theta, known_params);
  
  arma::vec att1(m);
  arma::mat Ptt1(m, m);
  ekf_update_step(0, y.col(0), a1, P1, att1, Ptt1);
  
  arma::uvec nonzero = arma::find(Ptt1.diag() > 0);
  arma::mat L = psd_chol(Ptt1);
  std::normal_distribution<> normal(0.0, 1.0);
  for (unsigned int i = 0; i < nsim; i++) {
    
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    
    alpha.slice(i).col(0) = att1 + L * um;
    
  }
  
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec normalized_weights(nsim);
  double loglik = 0.0;
  
  if(arma::is_finite(y(0))) {
    weights.col(0) = log_obs_density(0, alpha);
    
    for (unsigned int i = 0; i < nsim; i++) {
      weights(i, 0) +=  dmvnorm(alpha.slice(i).col(0), a1, P1, false, true) -
        dmvnorm(alpha.slice(i).col(0), att1, L, true, true);
    }
    
    
    double max_weight = weights.col(0).max();
    weights.col(0) = arma::exp(weights.col(0) - max_weight);
    double sum_weights = arma::accu(weights.col(0));
    
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -std::numeric_limits<double>::infinity();
    }
    loglik = max_weight + std::log(sum_weights / nsim);
  } else {
    weights.col(0).ones();
    normalized_weights.fill(1.0 / nsim);
  }
  
  for (unsigned int t = 0; t < n; t++) {
    
    arma::vec r(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    
    indices.col(t) = stratified_sample(normalized_weights, r, nsim);
    
    arma::mat att(m, nsim);
    arma::cube Ptt(m, m, nsim);
    arma::mat alphatmp(m, nsim);
    
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp.col(i) = alpha.slice(indices(i, t)).col(t);
      arma::mat Rt = R_fn(t,  alphatmp.col(i), theta, known_params, known_tv_params);
      arma::mat Pt = Rt * Rt.t();
      arma::vec at = T_fn(t, alphatmp.col(i), theta, known_params, known_tv_params);
      arma::vec tmp(m);
      ekf_update_step(t + 1, y.col(t + 1), at, Pt, tmp, Ptt.slice(i));
      att.col(i) = tmp;
      Ptt.slice(i) = psd_chol(Ptt.slice(i));
    }
    
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec um(m);
      for(unsigned int j = 0; j < m; j++) {
        um(j) = normal(engine);
      }
      alpha.slice(i).col(t + 1) = att.col(i) + Ptt.slice(i) * um;
    }
    if ((t < (n - 1)) && arma::is_finite(y(t + 1))) {
      weights.col(t + 1) = log_obs_density(t + 1, alpha);
      for (unsigned int i = 0; i < nsim; i++) {
        arma::mat Rt = R_fn(t,  alphatmp.col(i), theta, known_params, known_tv_params);
        arma::mat RR = Rt * Rt.t();
        arma::vec mean = T_fn(t, alphatmp.col(i), theta, known_params, known_tv_params);
        weights(i, t + 1) +=  dmvnorm(alpha.slice(i).col(t + 1), mean, RR, false, true) -
          dmvnorm(alpha.slice(i).col(t + 1), att.col(i), Ptt.slice(i), true, true);
      }
      double max_weight = weights.col(t + 1).max();
      weights.col(t + 1) = arma::exp(weights.col(t + 1) - max_weight);
      double sum_weights = arma::accu(weights.col(t + 1));
      if(sum_weights > 0.0){
        normalized_weights = weights.col(t + 1) / sum_weights;
      } else {
        return -std::numeric_limits<double>::infinity();
      }
      loglik += max_weight + std::log(sum_weights / nsim);
    } else {
      weights.col(t + 1).ones();
      normalized_weights.fill(1.0/nsim);
    }
  }
  return loglik;
  
}

void nlg_ssm::ekf_update_step(const unsigned int t, const arma::vec y, 
  const arma::vec& at, const arma::mat& Pt, arma::vec& att, arma::mat& Ptt) const {
  
  arma::uvec na_y = arma::find_nonfinite(y);
  
  if (na_y.n_elem < p) {
    arma::mat Zg = Z_gn(t, at, theta, known_params, known_tv_params);
    Zg.rows(na_y).zeros();
    arma::mat HHt = H_fn(t, at, theta, known_params, known_tv_params);
    HHt = HHt * HHt.t();
    HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
    
    arma::mat Ft = Zg * Pt * Zg.t() + HHt;
    
    arma::mat cholF = arma::chol(Ft);
    
    arma::vec vt = y - Z_fn(t, at, theta, known_params, known_tv_params);
    vt.rows(na_y).zeros();
    
    arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
    arma::mat Kt = Pt * Zg.t() * inv_cholF * inv_cholF.t();
    att = at + Kt * vt;
    Ptt = Pt - Kt * Ft * Kt.t();
    
  } else {
    att = at;
    Ptt = Pt;
  } 
}

double nlg_ssm::log_signal_pdf(const arma::mat& alpha) const {
  
  double ll = dmvnorm(alpha.col(0), a1_fn(theta, known_params), 
    P1_fn(theta, known_params), false, true);
  
  if (arma::is_finite(y.col(0))) {
    ll += dmvnorm(y.col(0), Z_fn(0, alpha.col(0), theta, known_params, known_tv_params), 
      H_fn(0, alpha.col(0), theta, known_params, known_tv_params), true, true);
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    arma::vec mean = T_fn(t, alpha.col(t), theta, known_params, known_tv_params);
    arma::mat cov = R_fn(t, alpha.col(t), theta, known_params, known_tv_params);
    cov = cov * cov.t();
    ll += dmvnorm(alpha.col(t+1), mean, cov, false, true);
    if (arma::is_finite(y.col(t+1))) {
      ll += dmvnorm(y.col(t + 1), Z_fn(t + 1, alpha.col(t + 1), theta, known_params, known_tv_params), 
        H_fn(t + 1, alpha.col(t + 1), theta, known_params, known_tv_params), true, true);
      
    }
  }
  return ll;
  
}

