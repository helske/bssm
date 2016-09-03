#include "svm.h"

//general constructor
svm::svm(arma::vec y, arma::mat Z, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, arma::vec phi,
  arma::mat xreg, arma::vec beta, unsigned int seed, bool log_space) :
  ngssm(y, Z, T, R, a1, P1, phi, xreg, beta, 0, seed),
  nz_y(y), log_space(log_space) {
  
  nz_y(arma::find(abs(y) < 1e-4)).fill(1e-4);
}

svm::svm(arma::vec y, arma::mat Z, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, arma::vec phi,
  arma::mat xreg, arma::vec beta, unsigned int seed) :
  ngssm(y, Z, T, R, a1, P1, phi, xreg, beta, 0, seed),
  nz_y(y), log_space(false) {
  
  nz_y(arma::find(abs(y) < 1e-4)).fill(1e-4);
}

double svm::proposal(const arma::vec& theta, const arma::vec& theta_prop) {
  double q = 0.0;
  if (log_space) {
    q = theta_prop(1) - theta(1) + theta_prop(2) - theta(2);
  }
  return q;
}

void svm::update_model(arma::vec theta) {
  
  if (log_space) {
    T(0, 0, 0) = theta(0);
    R(0, 0, 0) = exp(theta(1));
    compute_RR();
    P1(0, 0) = R(0, 0, 0) * R(0, 0, 0) / (1 - theta(0) * theta(0));
    phi.fill(exp(theta(2)));
  } else {
    T(0, 0, 0) = theta(0);
    R(0, 0, 0) = theta(1);
    compute_RR();
    P1(0, 0) = theta(1) * theta(1) / (1 - theta(0) * theta(0));
    phi.fill(theta(2));
  }
  if(xreg.n_cols > 0) {
    beta = theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1);
    compute_xbeta();
  }
  
}

arma::vec svm::get_theta(void) {
  arma::vec theta(3 + xreg.n_cols);
  
  if (log_space) {
    theta(0) = T(0, 0, 0);
    theta(1) = log(R(0, 0, 0));
    theta(2) = log(phi(0));
  } else {
    theta(0) = T(0, 0, 0);
    theta(1) = R(0, 0, 0);
    theta(2) = phi(0);
  }
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

//bootstrap filter
double svm::bootstrap_filter(unsigned int nsim, arma::cube& alphasim, arma::vec& V) {
  
  
  std::normal_distribution<> normal(0.0, 1.0);
  
  for (unsigned int i = 0; i < nsim; i++) {
    alphasim(0,0,i) = a1(0) + sqrt(P1(0)) * normal(engine);
  }
  double w = 0;
  if (arma::is_finite(ng_y(0))) {
    V = pyt2(0, arma::vectorise(alphasim.tube(0,0)));
    double maxV = V.max();
    V = exp(V - maxV);
    w = maxV + log(sum(V)) - log(nsim);
    V /= sum(V);
  } else {
    V.fill(1.0/nsim);
  }
  
  arma::mat ind(n, nsim);
  std::discrete_distribution<> sample(V.begin(), V.end());
  arma::vec alphatmp(nsim);
  if ((1.0/arma::sum(arma::square(V))) < (nsim*0.75)) {
    for (unsigned int i = 0; i < nsim; i++) {
      ind(0, i) = sample(engine);
      V(i) = 1.0/nsim;
    }
  } else {
    for (unsigned int i = 0; i < nsim; i++) {
      ind(0, i) = i;
    }
  }
  for (unsigned int i = 0; i < nsim; i++) {
    alphatmp(i) = alphasim(0, 0, ind(0, i));
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    for (unsigned int i = 0; i < nsim; i++) {
      alphasim(0, t + 1, i) = T(0) * alphatmp(i) + 
        R(0) * normal(engine);
    }
    if (arma::is_finite(ng_y(t + 1))) {
      V = log(V) + pyt2(t + 1, arma::vectorise(alphasim.tube(0,t + 1)));
      double maxV = V.max();
      V = exp(V - maxV);
      w += maxV + log(sum(V));// - log(nsim);
      V /= sum(V);
    } else {
      V.fill(1.0/nsim);
    }
    if (t < (n - 2) && (1.0/arma::sum(arma::square(V))) < (nsim*0.75)) {
      std::discrete_distribution<> sample(V.begin(), V.end());
      for (unsigned int i = 0; i < nsim; i++) {
        ind(t+1, i) = sample(engine);
        V(i) = 1.0/nsim;
      }
    } else {
      for (unsigned int i = 0; i < nsim; i++) {
        ind(t+1, i) = i;
      }
    }
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp(i) = alphasim(0, t+1, ind(t+1, i));
    }
    
  }
  
  for (int t = n - 2; t >= 0; t--) {
    arma::vec alphatmp = arma::vectorise(alphasim.tube(0, t));
    for (unsigned int i = 0; i < nsim; i++) {
      alphasim(0, t, i) = alphatmp(ind(t,i));
    }
  }
  return w;
}
// only for SV
double svm::gap_filter(unsigned int nsim, arma::cube& alphasim, arma::vec& V, arma::vec& init_signal) {
  
  
  //auxiliary particle filter
  
  std::normal_distribution<> normal(0.0, 1.0);
  
  double s = sqrt(P1(0));
  for (unsigned int i = 0; i < nsim; i++) {
    alphasim(0, 0, i) = a1(0) + s * normal(engine);
  }
  
  double w = 0;
  
  if (arma::is_finite(ng_y(0))) {
    V = pyt(0, alphasim);
    double maxV = V.max();
    V = exp(V - maxV);
    w = maxV + log(sum(V)) - log(nsim);
    V /= sum(V);
  } else {
    V.ones();
  }
  
  arma::mat ind(n, nsim);
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    arma::vec apred = T(0) * arma::vectorise(alphasim.tube(0,t));
    arma::vec predV = pyt2(t + 1, apred) + log(V); //!! V?
    double maxV = predV.max();
    predV = exp(predV - maxV);
    predV /= sum(predV);
    
    arma::vec alphatmp(nsim);
    std::discrete_distribution<> sample(predV.begin(), predV.end());
    for (unsigned int i = 0; i < nsim; i++) {
      ind(t, i) = sample(engine);
      alphatmp(i) = alphasim(0, t, ind(t, i));
    }
    apred = T(0) * alphatmp;
    for (unsigned int i = 0; i < nsim; i++) {
      alphasim(0, t + 1, i) = T(0) * alphatmp(i) + 
        R(0) * normal(engine);
    }
    if (arma::is_finite(ng_y(t + 1))) {
      V = pyt(t + 1, alphasim) - pyt2(t + 1, apred);
      double maxV = V.max();
      V = exp(V - maxV);
      w += maxV + log(sum(V)) - log(nsim);
      V /= sum(V);
    } else {
      V.ones();
    }
  }
  
  for (int t = n - 2; t >= 0; t--) {
    arma::vec alphatmp = arma::vectorise(alphasim.tube(0, t));
    for (unsigned int i = 0; i < nsim; i++) {
      alphasim(0, t, i) = alphatmp(ind(t,i));
    }
  }
  return w;
}

//bootstrap filter with initial value simulation from approximating model
// double svm::gap_filter(unsigned int nsim, arma::cube& alphasim, arma::vec& V, arma::vec& init_signal) {
//   
//   arma::mat att(m, n);
//   arma::cube Ptt(m, m, n);
//   arma::mat at(m, n+1);
//   arma::cube Pt(m, m, n+1);
//   double ll_approx = approx(init_signal, 100, 1e-8);
//   filter(at, att, Pt, Ptt, distribution != 0);
//   
//   arma::mat L(m, m);
//   L = arma::chol(Ptt.slice(0), "lower");
//   
//   std::normal_distribution<> normal(0.0, 1.0);
//   
//   for (unsigned int i = 0; i < nsim; i++) {
//     arma::vec um(m);
//     for(unsigned int j = 0; j < m; j++) {
//       um(j) = normal(engine);
//     }
//     alphasim.slice(i).col(0) = att.col(0) + L * um;
//   }
//   arma::uvec nonzero = arma::find(P1.diag() > 0);
//   arma::mat L_P1(m, m, arma::fill::zeros);
//   if (nonzero.n_elem > 0) {
//     L_P1.submat(nonzero, nonzero) =
//       arma::chol(P1.submat(nonzero, nonzero), "lower");
//   }
//   double w = 0;
//   if (arma::is_finite(ng_y(0))) {
//     V = pyt(0, alphasim) + dmvnorm1(alphasim.tube(arma::span::all, arma::span(0)),
//       a1, L_P1, true, true)
//     - dmvnorm1(alphasim.tube(arma::span::all, arma::span(0)),
//       att.col(0), L, true, true);
//     double maxV = V.max();
//     V = exp(V - maxV);
//     w = maxV + log(sum(V)) - log(nsim);
//     V /= sum(V);
//   } else {
//     V.ones();
//   }
//   for (unsigned int t = 0; t < (n - 1); t++) {
//     std::discrete_distribution<> sample(V.begin(), V.end());
//     arma::cube alphatmp(m, n, nsim);
//     for (unsigned int i = 0; i < nsim; i++) {
//       alphatmp(arma::span::all, arma::span(0, t), arma::span(i)) =
//         alphasim(arma::span::all, arma::span(0, t), arma::span(sample(engine)));
//     }
//     alphasim(arma::span::all, arma::span(0, t), arma::span::all) =
//       alphatmp(arma::span::all, arma::span(0, t), arma::span::all);
//     
//     L = arma::chol(Pt.slice(t + 1), "lower");
//     
//     for (unsigned int i = 0; i < nsim; i++) {
//       arma::vec um(m);
//       for(unsigned int j = 0; j < m; j++) {
//         um(j) = normal(engine);
//       }
//       alphasim.slice(i).col(t + 1) = T(0) * alphasim.slice(i).col(t) +
//         L * um;
//     }
//     if (arma::is_finite(ng_y(t + 1))) {
//       V = pyt(t + 1, alphasim) + 
//         dmvnorm2(alphasim.tube(arma::span::all, arma::span(t + 1)),
//         alphasim.tube(arma::span::all, arma::span(t)),
//         R.slice(Rtv * t), true, true, T.slice(Ttv * t))
//       - dmvnorm2(alphasim.tube(arma::span::all, arma::span(t + 1)),
//         alphasim.tube(arma::span::all, arma::span(t)),
//         L, true, true,T.slice(Ttv * t));
//       double maxV = V.max();
//       V = exp(V - maxV);
//       w += maxV + log(sum(V)) - log(nsim);
//       V /= sum(V);
//     } else {
//       V.ones();
//     }
//   }
//   
//   return w;
// }


//compute log-weights
arma::vec svm::pyt2(const unsigned int t, const arma::vec& alphasim) {
  
  arma::vec V(alphasim.n_elem);
  
  if (arma::is_finite(ng_y(t))) {
    for (unsigned int i = 0; i < alphasim.n_elem; i++) {
      V(i) = -0.5 * (LOG2PI + 2.0 * log(phi(0)) + alphasim(i) + 
        pow((ng_y(t) - xbeta(t))/phi(0), 2) * exp(-alphasim(i)));
      
    }
  } else {
    V.fill(-arma::datum::inf);
  }
  
  return V;
}

//bootstrap filter
double svm::bootstrap_loglik(unsigned int nsim, double ess_treshold) {
  
  arma::vec alphasim(nsim);
  arma::vec V(nsim);
  std::normal_distribution<> normal(0.0, 1.0);
  
  for (unsigned int i = 0; i < nsim; i++) {
    alphasim(i) = a1(0) + sqrt(P1(0)) * normal(engine);
  }
  double w = 0;
  if (arma::is_finite(ng_y(0))) {
    V = pyt2(0, alphasim);
    double maxV = V.max();
    V = exp(V - maxV);
    w = maxV + log(sum(V)) - log(nsim);
    V /= sum(V);
  } else {
    V.fill(1.0/nsim);
  }
  
  std::discrete_distribution<> sample(V.begin(), V.end());
  arma::vec ind(nsim);
  arma::vec alphatmp(nsim);
  if ((1.0/arma::sum(arma::square(V))) < (nsim*ess_treshold)) {
    for (unsigned int i = 0; i < nsim; i++) {
      ind(i) = sample(engine);
      V(i) = 1.0/nsim;
    }
  } else {
    for (unsigned int i = 0; i < nsim; i++) {
      ind(i) = i;
    }
  }
  for (unsigned int i = 0; i < nsim; i++) {
    alphatmp(i) = alphasim(ind(i));
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    for (unsigned int i = 0; i < nsim; i++) {
      alphasim(i) = T(0) * alphatmp(i) + 
        R(0) * normal(engine);
    }
    if (arma::is_finite(ng_y(t + 1))) {
      V = log(V) + pyt2(t + 1, alphasim);
      double maxV = V.max();
      V = exp(V - maxV);
      w += maxV + log(sum(V));// - log(nsim);
      V /= sum(V);
    } else {
      V.fill(1.0/nsim);
    }
    if (t < (n - 2) && (1.0/arma::sum(arma::square(V))) < (nsim*ess_treshold)) {
      std::discrete_distribution<> sample(V.begin(), V.end());
      for (unsigned int i = 0; i < nsim; i++) {
        ind(i) = sample(engine);
        V(i) = 1.0/nsim;
      }
    } else {
      for (unsigned int i = 0; i < nsim; i++) {
        ind(i) = i;
      }
    }
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp(i) = alphasim(ind(i));
    }
    
  }
  return w;
}


//bootstrap filter
double svm::bootstrap_filter2(unsigned int nsim, arma::cube& alphasim, arma::mat& V, arma::umat& ind,
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

// //bootstrap filter
// double svm::bootstrap_filter2(unsigned int nsim, arma::cube& alphasim, arma::mat& V, arma::umat& ind) {
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