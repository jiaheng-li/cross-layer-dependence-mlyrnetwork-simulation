#pragma once
#include <algorithm>
#include <vector>
#include <omp.h>
#include <iostream>
#include <stdlib.h>
#include <Rcpp.h> 
#include <RcppArmadillo.h>
using namespace std;


bool contains(vector<string>& vec, string value) {
    if (find(vec.begin(), vec.end(), value) != vec.end()) {
        return true;
    }
    else {
        return false;
    }
}

bool is_in(vector<int>& vec, int value) {
    if (find(vec.begin(), vec.end(), value) != vec.end()) {
        return true;
    }
    else {
        return false;
    }
}

int get_int(double u, int m) {
    for (int i = (m - 1); i >= 0; --i) {
        if (u >= ((double)i / m)) {
            return i;
        }
    }
    return -1;
}

void restart_vector(vector<double>& vec) {
    for (int i = 0; i < vec.size(); ++i) {
        vec.at(i) = 0.0;
    }
}

bool is_in_ch(arma::vec vec, arma::mat M) {
    Rcpp::NumericMatrix pvec = Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(vec));
    Rcpp::NumericMatrix Mmat = Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(M));
    cout << "\nmade it to estimation check point 4.3.0.0\n";
    Rcpp::Environment package_env("package:ergm");
    cout << "\nmade it to estimation check point 4.3.0.1\n";
    Rcpp::Function check_in_ch = package_env["is.inCH"];
    cout << "\nmade it to estimation check point 4.3.0.2\n";
    Rcpp::LogicalVector check = check_in_ch(Rcpp::transpose(pvec), Mmat);
    bool check_;
    check_ = check[0];
    return(check_);
}

double norm2(vector<double>& vec_) {
    double val = 0;
    for (int p = 0; p < vec_.size(); ++p) {
        val += pow(vec_[p], 2);
    }
    val = sqrt(val);
    return val;
}