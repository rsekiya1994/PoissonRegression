#ifndef _LinearFit_
#define _LinearFit_


#include <vector>
#include <iostream>
#include <numeric>
#include <iterator>
#include "mathfunction.hh"

class LinearFit{
public:
  LinearFit();
  ~LinearFit(){};

  template <class InputIterator1, class InputIterator2>
  void SetDataPoints(InputIterator1 x_first,
		     InputIterator1 x_last,
		     InputIterator2 y_first);
  void SetDataPoints(double x, double y);
  template <class InputIterator>
  void SetDataErrorY(InputIterator first,
		     InputIterator last);
  void LeastSquaresMethod(std::string option = "");

  void GetParameters(double &p0, double &p1){ p0 = p0_; p1 = p1_;};
  void PoissonRegression();
  double GetLowerCL(double count, double prob = 0.685, int n = 1);
  double GetUpperCL(double count, double prob = 0.685, int n = 1);
  void SetParameter(double p0, double p1);
private:
  int num_points = 0;
  std::vector<double> data_y;
  std::vector<double> data_x;
  std::vector<double> error_y;
  std::vector<std::pair<double, double> > error_low_high; // for poisson regression// low -> first, high -> second 
  // std::array<std::array<double, 2>, 2> U;
  std::array<double, 2> v;
  // double Generate_U(int alpha, int beta);
  // double Generate_v(int alpha);

  double p0_;
  double p1_;
  double p0_e;
  double p1_e;

  double p0_init = 0;
  double p1_init = 1;
  double Poisson(double x, double lambda){
    return std::exp(-lambda) * std::pow(lambda, x) / std::tgamma(x + 1);
  }
  
  template <class InputIterator>
  void LeastSquaresMethodSolver(InputIterator x_first,
				InputIterator x_last,
				InputIterator y_first,
				InputIterator y_last,
				double &p0, double &p1);
  template <class InputIterator>
  void PoissonRegSolver(InputIterator x_first,
			InputIterator x_last,
			InputIterator y_first,
			InputIterator y_last,
			double &p0, double &p1);

  // --- For poisson regression --- //
  void ErrorAssign();
  // ---- Calculate Partial Differencial value ---- //
  void GetError_PoissonReg();
  double p0_e_low;
  double p0_e_high;
  double p1_e_low;
  double p1_e_high;
  
};

LinearFit::LinearFit(){
  
}

template <class InputIterator1, class InputIterator2>
void LinearFit::SetDataPoints(InputIterator1 x_first,
			      InputIterator1 x_last,
			      InputIterator2 y_first){
  num_points = x_last - x_first;
  data_x.reserve(num_points);
  data_y.reserve(num_points);
  while(x_first != x_last){
    data_x.push_back( static_cast<double>(*x_first++) );
    data_y.push_back( static_cast<double>(*y_first++) );
  }
}

void LinearFit::SetDataPoints(double x, double y){
  num_points++;
  data_x.push_back(x);
  data_y.push_back(y);
}

void LinearFit::LeastSquaresMethod(std::string option){
  LeastSquaresMethodSolver(data_x.begin(), data_x.end(), data_y.begin(), data_y.end(), p0_, p1_);
  p1_e = 0;
  p0_e = 0;
  if(option != "Q"){
    std::cout << "---- Least Scuares Method Result ---- " << std::endl;
    std::cout << "p0 = " << p0_ << std::endl;
    std::cout << "p1 = " << p1_ << std::endl;
  }
}

template <class InputIterator>
void LinearFit::LeastSquaresMethodSolver(InputIterator x_first,
					 InputIterator x_last,
					 InputIterator y_first,
					 InputIterator y_last,
					 double &p0, double &p1){
  // num_points = x_last - x_first;
  double xk_yk  = std::inner_product(x_first, x_last, y_first, 0.0);
  double x_sum2 = std::accumulate(x_first, x_last, 0.0, [](double acc, const double &x){return acc + x * x;});
  double x_sum  = std::accumulate(x_first, x_last, 0.0, [](double acc, const double &x){return acc + x;});
  double y_sum  = std::accumulate(y_first, y_last, 0.0, [](double acc, const double &y){return acc + y;});
  p1 = (num_points * xk_yk - x_sum * y_sum) / (num_points * x_sum2 - x_sum * x_sum);
  p0 = (x_sum2 * y_sum - xk_yk * x_sum) / (num_points * x_sum2 - x_sum * x_sum);
}

template <class InputIterator>
void LinearFit::SetDataErrorY(InputIterator first, InputIterator last){
  int n_points = last - first;
  error_y.reserve(n_points);
  std::copy(first, last, std::back_inserter(error_y));
}

void LinearFit::ErrorAssign(){
  error_low_high.reserve(num_points);
  for(int i = 0 ; i < num_points ; i++){
    double count_now = data_y[i];
    double error_low = count_now - GetLowerCL(count_now);
    double error_up  = GetUpperCL(count_now) - count_now;
    // std::cout << count_now << "   " << count_now - error_low << " " << count_now + error_up << std::endl;
    error_low_high.push_back( {error_low, error_up} );
  }
}

void LinearFit::PoissonRegression(){
  ErrorAssign();
  // p0_ = 1;
  // p1_ = 1;

  PoissonRegSolver(data_x.begin(), data_x.end(), data_y.begin(), data_y.end(), p0_, p1_);
  p0_ += 1;
  GetError_PoissonReg();
  std::cout << "p0 = " << p0_ << ",  [" << p0_ - p0_e_low << ", " << p0_ + p0_e_high << "]" << std::endl;
  std::cout << "p1 = " << p1_ << ",  [" << p1_ - p1_e_low << ", " << p1_ + p1_e_high << "]" << std::endl;

}

void LinearFit::SetParameter(double p0, double p1){
  p0_init = p0;
  p1_init = p1;
}

template <class InputIterator>
void LinearFit::PoissonRegSolver(InputIterator x_first,
				 InputIterator x_last,
				 InputIterator y_first,
				 InputIterator y_last,
				 double &p0, double &p1){
  // num_points = x_last - x_first;
  // LeastSquaresMethodSolver(x_first, x_last, y_first, y_last, p0_old, p1_old);
  double p0_old = p0_init;
  double p1_old = p1_init;
  //p0_old -= 1;
  // std::cout << *x_first << " " << *y_first << std::endl;
  // std::cout << p0_old << " " << p1_old  << " ";
  int iteration_count = 0;
  while(1){
    double dp[2] = {0, 0};
    double ddp[2][2] = {{0, 0},{0, 0}};
    double W = 0;
    auto it_x = x_first;
    auto it_y = y_first;
    for(int i = 0 ; i < num_points ; i++){
      double x_i = *it_x++;
      double y_i = *it_y++;
      // std::cout << x_i << " " << y_i << std::endl;
      if(x_i <= - p0_old / p1_old){
	// std::cout << i << "  A" << std::endl;
	double expterm = std::exp(p0_old + p1_old * x_i);
	dp[0] += y_i - expterm;
	dp[1] += x_i * y_i - x_i * expterm;
	ddp[0][0] += - expterm;
	ddp[1][0] += - x_i * expterm;
	ddp[0][1] += - x_i * expterm;
	ddp[1][1] += - x_i * x_i * expterm;
	W += y_i * (p0_old + p1_old * x_i) - expterm;
      }else{
	//std::cout << i << "  B" << std::endl;
	double denominator = p0_old + p1_old * x_i + 1;
	dp[0]  += y_i / denominator - 1;
	dp[1]  += ( y_i  / denominator - 1 ) * x_i;
	ddp[0][0] -=  y_i / mathfunc::power(denominator, 2);
	ddp[0][1] -= (x_i * y_i) / mathfunc::power(denominator, 2);
	ddp[1][0] -= (y_i * x_i) / mathfunc::power(denominator, 2);
	ddp[1][1] -=  y_i * mathfunc::power(x_i / denominator, 2);
	W += y_i * std::log(denominator) - (denominator);
      }
    }
    double det = ddp[0][0] * ddp[1][1] - ddp[1][0] * ddp[0][1];
    double p0_new = p0_old - (ddp[1][1] * dp[0] - ddp[0][1] * dp[1]) / det;
    double p1_new = p1_old - (ddp[0][0] * dp[1] - ddp[1][0] * dp[0]) / det;
    if( mathfunc::power(p0_new - p0_old, 2) + mathfunc::power(p1_new - p1_old, 2) > 1e-15){
      p0_old = p0_new;
      p1_old = p1_new;
    }else{
      p0_old = p0_new;
      p1_old = p1_new;
      break;
    }
    //std::cout << dp[0] << " " << dp[1] << " : [p0, p1] = " << p0_new << " " << p1_new << "  W = " << W << std::endl;
    //std::cout << det << " "<< p0_new << " " << p1_new << " "<< W << std::endl;

    ++iteration_count;
    if(iteration_count > 50){
      std::cout << "Parameter Not Converged" << std::endl;
      break;
    }
  }

  p0 = p0_old;
  p1 = p1_old;
}


double LinearFit::GetLowerCL(double count, double prob ,int n){
  double deg = 2 * count;
  if(count <= 1.24) return 0;
  return mathfunc::chisquared_lower_limit(0.5 - prob / 2, deg) / (2 * n);
}

double LinearFit::GetUpperCL(double count, double prob,int n){
  double deg = 2 * count + 2;
  return mathfunc::chisquared_lower_limit(0.5 + prob / 2, deg) / (2 * n);
}

void LinearFit::GetError_PoissonReg(){
  double h = 0.001;
  std::vector<double> par_dp0_dy;//partial differenciate
  std::vector<double> par_dp1_dy;
  par_dp0_dy.reserve(num_points);
  par_dp1_dy.reserve(num_points);
  std::vector<double> x_copy;
  std::vector<double> y_copy;
  x_copy.reserve(num_points);
  y_copy.reserve(num_points);
  std::copy(data_x.begin(), data_x.end(), std::back_inserter(x_copy));
  std::copy(data_y.begin(), data_y.end(), std::back_inserter(y_copy));
  for(int i = 0 ; i < num_points ; i++){
    double p0_temp_p,  p0_temp_m;
    double p1_temp_p,  p1_temp_m;
    y_copy[i] = data_y[i];
    y_copy[i] += h;
    SetParameter(p0_, p1_);
    PoissonRegSolver(x_copy.begin(), x_copy.end(),
		     y_copy.begin(), y_copy.end(),
		     p0_temp_p, p1_temp_p);
    y_copy[i] -= 2 * h;
    PoissonRegSolver(x_copy.begin(), x_copy.end(),
		     y_copy.begin(), y_copy.end(),
		     p0_temp_m, p1_temp_m);
    y_copy[i] = data_y[i];
    // std::cout << "p0_temp_m : " << p0_temp_m << " "
    // 	      << "p1_temp_m : " << p1_temp_m << " "
    // 	      << "p0_temp_p : " << p0_temp_p << " "
    // 	      << "p1_temp_p : " << p1_temp_m << std::endl;
    par_dp0_dy.push_back( (p0_temp_p - p0_temp_m) / (2 * h) );
    par_dp1_dy.push_back( (p1_temp_p - p1_temp_m) / (2 * h) );
  }// <i>
  double dp0_low = 0;
  double dp1_low = 0;
  double dp0_high = 0;
  double dp1_high = 0;
  for(int i = 0 ; i < num_points ; i++){
    dp0_low  += mathfunc::power(par_dp0_dy[i] * error_low_high[i].first , 2);
    dp0_high += mathfunc::power(par_dp0_dy[i] * error_low_high[i].second, 2);
    dp1_low  += mathfunc::power(par_dp1_dy[i] * error_low_high[i].first , 2);
    dp1_high += mathfunc::power(par_dp1_dy[i] * error_low_high[i].second, 2);
  }
  p0_e_low  = std::sqrt(dp0_low);
  p1_e_low  = std::sqrt(dp1_low);
  p0_e_high = std::sqrt(dp0_high);
  p1_e_high = std::sqrt(dp1_high);
  std::cout << "dp0_low = " << p0_e_low << std::endl;
  std::cout << "dp1_low = " << p1_e_low << std::endl;
  std::cout << "dp0_high = " << p0_e_high << std::endl;
  std::cout << "dp1_high = " << p1_e_high << std::endl;
}

#endif
