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
  HH = 2 * exp(signal) / pow((nz_y - xbeta)/phi(0), 2);
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


// // only for SV
// double svm::gap_filter(unsigned int nsim, arma::cube& alphasim, arma::vec& V, arma::vec& init_signal) {
//   
//   std::normal_distribution<> normal(0.0, 1.0);
//   
//   double s = sqrt(P1(0));
//   for (unsigned int i = 0; i < nsim; i++) {
//     alphasim(0, 0, i) = a1(0) + s * normal(engine);
//   }
//   
//   double w = 0;
//   if (arma::is_finite(ng_y(0))) {
//     V = pyt(0, alphasim);
//     double maxV = V.max();
//     V = exp(V - maxV);
//     w = maxV + log(sum(V)) - log(nsim);
//     V /= sum(V);
//   } else {
//     V.ones();
//   }
//   
//   arma::vec Pt(nsim);
//   Pt.fill(arma::as_scalar(P1));
//   arma::vec at(nsim);
//   
//   for (unsigned int t = 0; t < (n - 1); t++) {
//     std::discrete_distribution<> sample(V.begin(), V.end());
//     arma::cube alphatmp(1, n, nsim);
//     //arma::vec Ptmp(nsim);
//     
//     for (unsigned int i = 0; i < nsim; i++) {
//       unsigned int ind = sample(engine);
//      // Ptmp(i) = Pt(ind);
//       alphatmp(arma::span::all, arma::span(0, t), arma::span(i)) = 
//         alphasim(arma::span::all, arma::span(0, t), arma::span(ind));
//     }
//     alphasim(arma::span::all, arma::span(0, t), arma::span::all) = 
//       alphatmp(arma::span::all, arma::span(0, t), arma::span::all);
//    // Pt = Ptmp;
//    
//   double mean_a = arma::mean(arma::vectorise(alphasim.tube(0, t)));
//   double pseudo_HH = 2.0 * phi(0) * phi(0) *
//     exp(mean_a) / pow(ng_y(t)-xbeta(t), 2);
//   double pseudo_y = mean_a + 1.0 - 0.5 * pseudo_HH;
//   
//     for (unsigned int i = 0; i < nsim; i++) {
//      
//       // double pseudo_HH = 2.0 * phi(0) * phi(0) *
//       //   exp(alphasim(0, t, i)) / pow(ng_y(t)-xbeta(t), 2);
//       // double pseudo_y = alphasim(0, t, i) + 1.0 - 0.5 * pseudo_HH;
//       // 
//       // at(i) = uv_filter2(pseudo_y, pseudo_HH,
//       //   T(0), RR(0), alphasim(0, t, i), Pt(i), zero_tol);
//       // 
//       
//       at(i) = uv_filter2(pseudo_y, pseudo_HH,
//         T(0), RR(0), alphasim(0, t, i), Pt(i), zero_tol);
//       
//       alphasim(0, t + 1, i) = at(i) + sqrt(Pt(i)) * normal(engine);
//     }
//     if (arma::is_finite(ng_y(t + 1))) {
//       V = pyt(t + 1, alphasim);
//       for (unsigned int i = 0; i < nsim; i++) {
//        V(i) += R::dnorm(alphasim(0, t + 1, i), T(0) * alphasim(0, t, i), R(0), 1) - 
//          R::dnorm(alphasim(0, t + 1, i), at(i), sqrt(Pt(i)), 1);
//       }
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
// 
//bootstrap filter with initial value simulation from approximating model
double svm::gap_filter(unsigned int nsim, arma::cube& alphasim, arma::vec& V, arma::vec& init_signal) {
  
  arma::mat att(m, n);
  arma::cube Ptt(m, m, n);
  arma::mat at(m, n+1);
  arma::cube Pt(m, m, n+1);
  double ll_approx = approx(init_signal, 100, 1e-8);
  filter(at, att, Pt, Ptt, distribution != 0);
  
  arma::mat L(m, m);
  L = arma::chol(Ptt.slice(0), "lower");
  
  std::normal_distribution<> normal(0.0, 1.0);
  
  for (unsigned int i = 0; i < nsim; i++) {
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    alphasim.slice(i).col(0) = att.col(0) + L * um;
  }
  arma::uvec nonzero = arma::find(P1.diag() > 0);
  arma::mat L_P1(m, m, arma::fill::zeros);
  if (nonzero.n_elem > 0) {
    L_P1.submat(nonzero, nonzero) =
      arma::chol(P1.submat(nonzero, nonzero), "lower");
  }
  double w = 0;
  if (arma::is_finite(ng_y(0))) {
    V = pyt(0, alphasim) + dmvnorm1(alphasim.tube(arma::span::all, arma::span(0)),
      a1, L_P1, true, true)
    - dmvnorm1(alphasim.tube(arma::span::all, arma::span(0)),
      att.col(0), L, true, true);
    double maxV = V.max();
    V = exp(V - maxV);
    w = maxV + log(sum(V)) - log(nsim);
    V /= sum(V);
  } else {
    V.ones();
  }
  for (unsigned int t = 0; t < (n - 1); t++) {
    std::discrete_distribution<> sample(V.begin(), V.end());
    arma::cube alphatmp(m, n, nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp(arma::span::all, arma::span(0, t), arma::span(i)) =
        alphasim(arma::span::all, arma::span(0, t), arma::span(sample(engine)));
    }
    alphasim(arma::span::all, arma::span(0, t), arma::span::all) =
      alphatmp(arma::span::all, arma::span(0, t), arma::span::all);
    
    L = arma::chol(Pt.slice(t + 1), "lower");
    
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec um(m);
      for(unsigned int j = 0; j < m; j++) {
        um(j) = normal(engine);
      }
      alphasim.slice(i).col(t + 1) = T(0) * alphasim.slice(i).col(t) +
        L * um;
    }
    if (arma::is_finite(ng_y(t + 1))) {
      V = pyt(t + 1, alphasim) + 
        dmvnorm2(alphasim.tube(arma::span::all, arma::span(t + 1)),
        alphasim.tube(arma::span::all, arma::span(t)),
        R.slice(Rtv * t), true, true, T.slice(Ttv * t))
      - dmvnorm2(alphasim.tube(arma::span::all, arma::span(t + 1)),
        alphasim.tube(arma::span::all, arma::span(t)),
        L, true, true,T.slice(Ttv * t));
      double maxV = V.max();
      V = exp(V - maxV);
      w += maxV + log(sum(V)) - log(nsim);
      V /= sum(V);
    } else {
      V.ones();
    }
  }
  
  return w;
}