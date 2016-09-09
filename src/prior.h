#ifndef PRIOR_H
#define PRIOR_H

#include "bssm.h"

class prior {

public:

  prior(List);
  std::string prior_type;
  virtual double pdf(double, int) = 0.0;
};

class uniform: public prior {

public:
  
  uniform(List);
  double pdf(double, int);
  
private:
  double min;
  double max;
  
};

class halfnormal: public prior {
  
public:
  
  halfnormal(List);
  double pdf(double, int);
  
private:
  double sd;
  
};

class normal: public prior {
  
public:
  
  normal(List);
  double pdf(double, int);
  
private:
  double mean;
  double sd;
  
};

#endif
