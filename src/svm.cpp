#include "svm.h"

//general constructor
svm::svm(arma::vec y, arma::mat Z, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, arma::vec phi,
  arma::mat xreg, arma::vec beta, unsigned int seed, double prior_sd) :
  ngssm(y, Z, T, R, a1, P1, phi, xreg, beta, 0, seed),
  nz_y(y), prior_sd(prior_sd) {
  
  nz_y(arma::find(abs(y) < 1e-4)).fill(1e-4);
}

svm::svm(arma::vec y, arma::mat Z, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, arma::vec phi,
  arma::mat xreg, arma::vec beta, unsigned int seed) :
  ngssm(y, Z, T, R, a1, P1, phi, xreg, beta, 0, seed),
  nz_y(y), prior_sd(100) {
  
  nz_y(arma::find(abs(y) < 1e-4)).fill(1e-4);
}


double svm::proposal(const arma::vec& theta, const arma::vec& theta_prop) {
  
  return R::dnorm(theta_prop(2),0, prior_sd, 1) - 
    R::dnorm(theta(2),0, prior_sd,1);
  
}

void svm::update_model(arma::vec theta) {
  
  T(0, 0, 0) = theta(0);
  R(0, 0, 0) = theta(1);
  compute_RR();
  P1(0, 0) = theta(1) * theta(1) / (1 - theta(0) * theta(0));
  phi.fill(theta(2));
  
  if(xreg.n_cols > 0) {
    beta = theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1);
    compute_xbeta();
  }
  
}

arma::vec svm::get_theta(void) {
  arma::vec theta(3 + xreg.n_cols);
  
  theta(0) = T(0, 0, 0);
  theta(1) = R(0, 0, 0);
  theta(2) = phi(0);
  
  if(xreg.n_cols > 0) {
    theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1) = beta;
  }
  return theta;
}

// compute new values of pseudo y and H given the signal
// and the new signal using Kalman smoothing
arma::vec svm::approx_iter(arma::vec& signal) {
  // new pseudo y and H
  HH = 2.0 * exp(signal) / pow((nz_y - xbeta)/phi(0), 2);
  y = signal + 1.0 - 0.5 * HH;
  // new signal
  
  // arma::mat alpha = arma::vectorise(fast_smoother(false));
  // arma::vec signal_new = alpha.row(0).t();
  
  // for (unsigned int t = 0; t < n; t++) {
  //   signal_new(t) = arma::as_scalar(Z.col(Ztv * t).t() * alpha.col(t));
  // }
  H = sqrt(HH);
  
  return arma::vectorise(fast_smoother(false));
}


// log[p(y | signal)]
double svm::logp_y(arma::vec& signal) {
  
  double logp = 0.0;
  
  for (unsigned int t = 0; t < n; t++) {
    if (arma::is_finite(y(t))) {
      logp -= 0.5 * (LOG2PI + 2.0 * log(phi(0)) + signal(t) + pow((ng_y(t) - xbeta(t))/phi(0), 2) * exp(-signal(t)));
    }
  }
  
  return logp;
}
//particle filter
double svm::particle_filter2(unsigned int nsim, arma::cube& alphasim, arma::mat& V, arma::umat& ind,
  arma::vec signal, double q) {
  
  Rcout<<"start"<<std::endl;
  double ll_approx = approx(signal, 50, 1e-12);
  
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
  double at = a1(0);
  double Pt = P1(0);
  
  double logU = 0.0;
  arma::vec Vnorm(nsim);
  //prediction is already a1
  // uv_filter2(y(0), HH(0), T(0), RR(0),
  //   at, Pt, zero_tol);
  if (arma::is_finite(ng_y(0))) {
    double F = Pt + HH(0);
    double v = y(0) - at;
    double K = Pt / F;
    at += K * v;
    Pt = (1.0 - K)*Pt;
    for (unsigned int i = 0; i < nsim; i++) {
      if (unif(engine) < q) {
        alphasim(0,0,i) = a1(0) + sqrt(P1(0)) * normal(engine);
      } else {
        alphasim(0,0,i) = at + sqrt(Pt) * normal(engine);
      }
      double pp = R::dnorm(alphasim(0,0,i), a1(0), sqrt(P1(0)), 0);
      V(i, 0) = R::dnorm(ng_y(0), xbeta(0), phi(0)*exp(alphasim(0,0,i)/2.0), 0) * pp /
      (q*pp + (1.0 - q)*R::dnorm(alphasim(0,0,i), at, sqrt(Pt), 0));
    }
    logU = log(arma::mean(V.col(0)));
  } else {
    // y(0) missing, at=a1 and Pt=P1
    for (unsigned int i = 0; i < nsim; i++) {
      alphasim(0,0,i) = a1(0) + sqrt(P1(0)) * normal(engine);
    }
    V.col(0).ones();
  }
  Vnorm = V.col(0) / arma::sum(V.col(0));
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    arma::vec r(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    
    ind.col(t) = stratified_sample(Vnorm, r, nsim);
    
    arma::vec alphatmp(nsim);
    
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp(i) = alphasim(0, t, ind(i, t));
    }
    
    Pt = RR(0);
    double F = Pt + HH(t+1);
    double K = Pt / F;
    Pt = (1.0 - K)*Pt;
    Rcout<<Pt<<std::endl;
    if (arma::is_finite(ng_y(t+1))) {
      for (unsigned int i = 0; i < nsim; i++) {
        // prediction
        at = T(0) * alphatmp(i);
        // update
        at += K * (y(t+1) - at);
        if (unif(engine) < q) {
          alphasim(0, t + 1, i) = T(0) * alphatmp(i) + R(0) * normal(engine);
        } else {
          alphasim(0, t + 1, i) = at + sqrt(Pt) * normal(engine);
        }
        double pp = R::dnorm(alphasim(0, t + 1, i), T(0) * alphatmp(i), R(0), 0);
        V(i, t + 1) = R::dnorm(ng_y(t + 1), xbeta(t + 1), phi(0) * exp(alphasim(0, t + 1, i) / 2.0), 0) * pp /
        (q*pp + (1.0-q)*R::dnorm(alphasim(0, t + 1, i), at, sqrt(Pt), 0));
      }
      
      logU += log(arma::mean(V.col(t + 1)));
    } else {
      // y(t+1) missing, proposals are same
      for (unsigned int i = 0; i < nsim; i++) {
        alphasim(0, t + 1, i) = T(0) * alphatmp(i) + R(0) * normal(engine);
      }
      V.col(t + 1).ones();
    }
    Vnorm = V.col(t + 1) / arma::sum(V.col(t + 1));
  }
  return logU;
}

// //particle filter
// double svm::particle_filter2(unsigned int nsim, arma::cube& alphasim, arma::mat& V, arma::umat& ind) {
//   
//   std::normal_distribution<> normal(0.0, 1.0);
//   
//   for (unsigned int i = 0; i < nsim; i++) {
//     alphasim(0,0,i) = a1(0) + sqrt(P1(0)) * normal(engine);
//   }
//   
//   double w = 0;
//   
//   arma::vec Vnorm(nsim);
//   if (arma::is_finite(ng_y(0))) {
//     V.col(0) = pyt2(0, arma::vectorise(alphasim.tube(0,0)));
//     double maxV = V.col(0).max();
//     Vnorm = exp(V.col(0) - maxV);
//     w = maxV + log(sum(Vnorm)) - log(nsim);
//     Vnorm = Vnorm / sum(Vnorm);
//   } else {
//     Vnorm.fill(1.0/nsim);
//   }
//   
//   
//   
//   for (unsigned int t = 0; t < (n - 1); t++) {
//     
//     std::uniform_real_distribution<> unif(0.0, 1.0);
//     
//     arma::vec r(nsim);
//     for (unsigned int i = 0; i < nsim; i++) {
//       r(i) = unif(engine);
//     }
//     
//     ind.col(t) = stratified_sample(Vnorm, r, nsim);
//     
//     arma::vec alphatmp(nsim);
//     
//     for (unsigned int i = 0; i < nsim; i++) {
//       alphatmp(i) = alphasim(0, t, ind(i, t));
//     }    
//     for (unsigned int i = 0; i < nsim; i++) {
//       alphasim(0, t + 1, i) = T(0) * alphatmp(i) + 
//         R(0) * normal(engine);
//     }
//     
//     if (arma::is_finite(ng_y(t + 1))) {
//       V.col(t+1) = pyt2(t + 1, arma::vectorise(alphasim.tube(0,t + 1)));
//       double maxV = V.col(t+1).max();
//       Vnorm = exp(V.col(t+1) - maxV);
//       w += maxV + log(sum(Vnorm)) - log(nsim);
//       Vnorm /= sum(Vnorm);
//     } else {
//       V.fill(1.0/nsim);
//     }
//     
//   }
//   return w;
// }