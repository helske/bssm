#include "summary.h"


void summary(const arma::cube& x, arma::mat& mean_x, arma::cube& cov_x) {
  
  cov_x.zeros();
  mean_x.zeros();
  for(unsigned int i = 0; i < x.n_slices; i++) {
    arma::mat diff = x.slice(i) - mean_x;
    mean_x += diff / (i + 1);
    arma::mat diff2 = (x.slice(i) - mean_x).t();
    for (unsigned int t = 0; t < x.n_cols; t++) {
      cov_x.slice(t) +=  diff.col(t) * diff2.row(t);
    }
  }
  cov_x /= x.n_slices;
}

void weighted_summary(const arma::cube& x, arma::mat& mean_x, arma::cube& cov_x, 
  const arma::vec& weights) {
  
  cov_x.zeros();
  mean_x.zeros();
  double cumsumw = 0;
  for(unsigned int i = 0; i < x.n_slices; i++) {
    if(weights(i) > 0) {
      double tmp = weights(i) + cumsumw;
      arma::mat diff = x.slice(i) - mean_x;
      mean_x += diff * weights(i) / tmp;
      arma::mat diff2 = (x.slice(i) - mean_x).t();
      for (unsigned int t = 0; t < x.n_cols; t++) {
        cov_x.slice(t) +=  weights(i) * diff.col(t) * diff2.row(t);
      }
      cumsumw = tmp;
    }
  }
  cov_x = cov_x / cumsumw;
  
}


void filter_summary(const arma::cube& alpha, arma::mat& at, arma::mat& att, 
  arma::cube& Pt, arma::cube& Ptt, arma::mat weights) {
  
  at.zeros();
  att.zeros();
  
  for (unsigned int t = 0; t < alpha.n_cols - 1; t++) {
    weights.col(t) /= arma::accu(weights.col(t));
    for (unsigned int i = 0; i < alpha.n_slices; i++) {
      att.col(t) += alpha.slice(i).col(t) * weights(i, t);
      at.col(t) += alpha.slice(i).col(t);
    }
  }
  weights.col(alpha.n_cols - 1) /= arma::accu(weights.col(alpha.n_cols - 1));
  for (unsigned int i = 0; i < alpha.n_slices; i++) {
    at.col(alpha.n_cols - 1) += alpha.slice(i).col(alpha.n_cols - 1);
  }
  
  at /= alpha.n_slices;
  
  Pt.zeros();
  Ptt.zeros();
  for (unsigned int t = 0; t < alpha.n_cols - 1; t++) {
    for(unsigned int i = 0; i < alpha.n_slices; i++) {
      Pt.slice(t) += 
        (alpha.slice(i).col(t) - at.col(t)) * (alpha.slice(i).col(t) - at.col(t)).t();
      Ptt.slice(t) += weights(i, t) * 
        (alpha.slice(i).col(t) - att.col(t)) * (alpha.slice(i).col(t) - att.col(t)).t();
      
    }
  }
  double t = alpha.n_cols - 1;
  for(unsigned int i = 0; i < alpha.n_slices; i++) {
    Pt.slice(t) += 
      (alpha.slice(i).col(t) - at.col(t)) * (alpha.slice(i).col(t) - at.col(t)).t();
  }
  Pt /= alpha.n_slices;
}


void sample_or_summarise(
    bool sample,
    const unsigned int method, 
    arma::cube& alpha, 
    const arma::vec& weights, 
    const arma::umat& indices,
    arma::mat& sampled_alpha, 
    arma::mat& alphahat, 
    arma::cube& Vt,  
    sitmo::prng_engine& engine) {
  
  if (method != 3) { // SPDK does not use this
    filter_smoother(alpha, indices);
  }
  if (sample) {
    std::discrete_distribution<unsigned int> sample(weights.begin(), weights.end());
    sampled_alpha = alpha.slice(sample(engine));
  } else {
    weighted_summary(alpha, alphahat, Vt, weights);
  }
  
}
