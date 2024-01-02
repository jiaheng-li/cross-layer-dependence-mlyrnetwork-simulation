#pragma once
#include <algorithm>
#include <RcppArmadillo.h>
#include <vector>
#include <omp.h>
#include <iostream>
#include <stdlib.h>
#include "helper_functions.h" 
#include "simulation_class.h" 
#include "model.h" 
#include "graph_class.h" 

using namespace std;
using namespace arma;
typedef std::vector<double> stdvec;


class est_ml_Hway {

public:
    sim_ml simulation_ml;
    mlnet obs_net_ml;
    vec theta_est;
    int model_dim;
    int H;
    vector<int> all_interaction_layer;
    vector<int> combination;
    vector<vector <int> > selected_layer;
    
    

public:
    est_ml_Hway(int nsamp, int burn, int intv, int mdim, vector<string> mterms,
        int N, int K, int highest_order, double random_seed, double gy)
        : simulation_ml(nsamp, burn, intv, mdim,  mterms, N, K, random_seed, gy),
        obs_net_ml(N, K),
        theta_est(mdim)
    {   
        model_dim = mdim;
        H = highest_order;
    }

    int get_model_dim() {
        return simulation_ml.get_model_dim();
    }

    vector<double> get_theta() {
        return conv_to< stdvec >::from(theta_est);
    }

    vector<double> vec_to_stdvec(vec& vec_) {
        return conv_to< stdvec >::from(vec_);
    }

    vec stdvec_to_vec(vector<double>& vec_) {
        return conv_to< vec >::from(vec_);
    }


    int compute_change_stats(int i, int j, int k, vector <int>& layers) {
        bool f = true;
        for (int ii = 0; ii < layers.size(); ii++) {
            if (!obs_net_ml.is_edge(i, j, layers[ii]) && layers[ii] != k) {
                f = false;
                break;
            }
        }
        if (f) return 1;
        else return 0;
    }

    int fact(int n)
    {
        if (n == 0) return 1;
        if (n == 1) return 1;
        return n * fact(n - 1);
    }


    void select_layer(int offset, int k) {
        if (k == 0) {
            selected_layer.push_back(combination);
            return;
        }
        for (int i = offset; i <= all_interaction_layer.size() - k; ++i) {
            combination.push_back(all_interaction_layer[i]);
            select_layer(i + 1, k - 1);
            combination.pop_back();
        }
    }

    int nCr(int n, int r)
    {
        return fact(n) / (fact(r) * fact(n - r));
    }

    void compute_initial_estimate() {
        double obs_val;
        for (int e = 0; e < obs_net_ml.layer_count(); ++e) {
            all_interaction_layer.push_back(e);
        }
        vector<double>  change_stat;   /// Change statistics for each edge
        change_stat.resize(get_model_dim());   //need to be changed to the number of dimensions);
        

        vector<vector<double> > theta;
        theta.resize(2);
        theta[0].resize(get_model_dim());
        theta[1].resize(get_model_dim());

        vector<double> gradient;
        gradient.resize(get_model_dim());

        // Compute change statistics for each edge



        // Estimate initial theta using initial guess of zero vector 
        for (int p = 0; p < get_model_dim(); ++p) {
            theta[0][p] = 0;
            theta[1][p] = 0;
        }

        int s = 0;
        bool conv_flag = false;
        double exp_val, scale_val, proposed_step_sum, gamma;
        double inner_prod = 0.0;
        int max_dim = 0;
        int para_dim = 0;
        
        //Rcpp::Rcout << "\n  start of the iteration " << "\n";
        for (int iter_num = 0; iter_num < 30000; ++iter_num) { /// Maximum number of iterations for gradient descent algm.
            // Compute gradient for edge
            for (int p = 0; p < get_model_dim(); ++p) {
                gradient[p] = 0;
            }
            for (int i = 0; i < (obs_net_ml.node_count() - 1); ++i) {
                for (int j = (i + 1); j < obs_net_ml.node_count(); ++j) {
                    for (int k = 0;k < obs_net_ml.layer_count();++k) {
                        for (int p = 0; p < get_model_dim();++p) {
                            change_stat[p] = 0;
                        }
                        bool flag_for1 = false;
                        for (int l = 0;l < obs_net_ml.layer_count();++l) {
                            if (l != k && obs_net_ml.is_edge(i, j, l)) {
                                flag_for1 = true;
                                break;
                            }
                        }
                        if (!flag_for1) continue;

                        inner_prod = 0.0;
                        para_dim = 0;
                        for (int h = 1; h <= H; ++h) {
                            select_layer(0, h);
                            for (auto ele : selected_layer) {
                                if (binary_search(ele.begin(), ele.end(), k)) {
                                    change_stat[para_dim] = compute_change_stats(i, j, k, ele); 
                                    inner_prod += theta[0][para_dim] * change_stat[para_dim];
                                    ++para_dim;
                                    continue;
                                }
                                else {
                                    ++para_dim;
                                    continue;
                                }
                            }
                            
                            // Re-initialize for next use
                            s = selected_layer.size();
                            for (int num = 0; num < s;++num) {
                                selected_layer.pop_back();
                            }


                            s = combination.size();
                            for (int num = 0; num < s;++num) {
                                combination.pop_back();
                            }
                           
                            
                        }

                        exp_val = exp(inner_prod);

                        if (obs_net_ml.is_edge(i, j, k)) { 
                            obs_val = 1;
                        }
                        else {
                            obs_val = 0;
                        }

                        scale_val = (exp_val / (1 + exp_val)) - obs_val;

                        for (int d = 0; d < get_model_dim(); ++d) {
                            gradient[d] += change_stat[d] * scale_val;
                            //Rcpp::Rcout << "\n  gradient is " << gradient[d] << ", " << "\n";
                        }
                    }

                }
                //Rcpp::Rcout << "\n  end of the iteration of one dyad" << "\n";
            }
            
            // Compute gamma scaling parameter and step increment
            proposed_step_sum = 0;
            for (int p = 0; p < get_model_dim(); ++p) {
                proposed_step_sum += pow(gradient[p], 2);
            }
            gamma = 1 / (pow(obs_net_ml.node_count(), 2));


            for (int p = 0; p < get_model_dim(); ++p) {
                theta[1][p] = theta[0][p];  
                theta[0][p] -= gamma * gradient[p];
            }


            if (sqrt(proposed_step_sum) < 1e-4 * get_model_dim()) {
                Rcpp::Rcout << "\n    Took " << iter_num << " iterations to converge to initial theta = (";
                for (int p = 0; p < get_model_dim(); ++p) {
                    Rcpp::Rcout << theta[0][p];
                    if (p == (get_model_dim() - 1)) {
                        Rcpp::Rcout << ").\n" << flush;
                    }
                    else {
                        Rcpp::Rcout << ", ";
                    }
                }
                conv_flag = true;
                break;
            }
        }
        Rcpp::Rcout << "\n  end of the iteration " << "\n";
        if (!conv_flag) {
            Rcpp::Rcout << "\n    Initial estimation procedure failed to converge. Starting estimate may be unstable.";
        }
        for (int p = 0; p < get_model_dim(); ++p) {
            theta_est[p] = theta[0][p];
        }
    }
};



class est_ml_3way {

public:
    sim_ml simulation_ml;
    mlnet obs_net_ml;
    vec theta_est;
    vec obs_vec;
    vec obs_vec_hold;
    vector<double> weights;
    mat information_matrix;
    int NR_max_iter;  /// For Newton Raphson method
    int MCMLE_max_iter;
    int model_dim;
    double NR_tol;
    bool check_chull;
    int cs3; // change statistic for 3-way interactions.

public:
    est_ml_3way(int nsamp, int burn, int intv, int mdim, vector<string> mterms,  /// Use corsslayer_est as the cross-layer parameter name for estimation
        int N, int K, double random_seed,
        int NR_max, double NRtol, int MCMLE_max, bool check_ch, double gy)
        : simulation_ml(nsamp, burn, intv, mdim, mterms, N, K, random_seed, gy),
        obs_net_ml(N, K),
        theta_est(mdim),
        obs_vec(mdim),
        obs_vec_hold(mdim),
        information_matrix(mdim, mdim)
    {
        NR_max_iter = NR_max;
        NR_tol = NRtol;
        MCMLE_max_iter = MCMLE_max;
        model_dim = mdim;
        check_chull = check_ch;
        weights.resize(nsamp);
    }

    int get_model_dim() {
        return simulation_ml.get_model_dim();
    }

    vector<double> get_theta() {
        return conv_to< stdvec >::from(theta_est);
    }

    vector<double> vec_to_stdvec(vec& vec_) {
        return conv_to< stdvec >::from(vec_);
    }

    vec stdvec_to_vec(vector<double>& vec_) {
        return conv_to< vec >::from(vec_);
    }


    void compute_change_stats(int i, int j, vector<vector<double> >& ch_stat, int k, int l) {
        ch_stat[k][l] = simulation_ml.m.change_stat_funs_ml[0](i, j, obs_net_ml, k, l);  /// Use cs_crosslayer_est for change stats of 2-layer co-ocurrences
    }

    void compute_initial_estimate() {
        double obs_val;

        vector<vector<double> >  change_stats;   /// Change statistics for each edge
        change_stats.resize(obs_net_ml.layer_count());
        for (int i = 0; i < obs_net_ml.layer_count(); ++i) {
            change_stats[i].resize(obs_net_ml.layer_count());
        }

        vector<vector<double> > theta;
        theta.resize(2);
        theta[0].resize(get_model_dim());
        theta[1].resize(get_model_dim());

        vector<double> gradient;
        gradient.resize(get_model_dim());

        // Compute change statistics for each edge



        // Estimate initial theta using initial guess of zero vector 
        for (int p = 0; p < get_model_dim(); ++p) {
            theta[0][p] = 0;
            theta[1][p] = 0;
        }

        vector<vector<double> >  theta_mat;
        theta_mat.resize(obs_net_ml.layer_count());
        for (int i = 0; i < obs_net_ml.layer_count(); ++i) {
            theta_mat[i].resize(obs_net_ml.layer_count());
        }


        bool conv_flag = false;
        double exp_val, scale_val, proposed_step_sum, gamma;
        double inner_prod = 0.0;
        vector<double> delta;
        delta.resize(get_model_dim());
        vector<vector<double> > delta_mat;
        delta_mat.resize(obs_net_ml.layer_count());
        for (int i = 0; i < obs_net_ml.layer_count(); ++i) {
            delta_mat[i].resize(obs_net_ml.layer_count());
        }
        for (int iter_num = 0; iter_num < 30000; ++iter_num) { /// Maximum number of iterations for gradient descent algm.
            // Compute gradient for edge
            for (int p = 0; p < get_model_dim(); ++p) {
                gradient[p] = 0;
            }
            bool flag;
            for (int i = 0; i < (obs_net_ml.node_count() - 1); ++i) {
                for (int j = (i + 1); j < obs_net_ml.node_count(); ++j) {

                   

                    for (int k = 0;k < obs_net_ml.layer_count();++k) {
                        bool flag_for1 = false;
                        for (int l = 0;l < obs_net_ml.layer_count();++l) {
                            if (l != k && obs_net_ml.is_edge(i, j, l)) {
                                flag_for1 = true;
                                break;
                            }
                        }
                        if (!flag_for1) continue;

                        flag = true;
                        for (int d = 0; d < get_model_dim(); ++d) {
                            delta[d] = 0;
                        }
                        for (int g = 0; g < obs_net_ml.layer_count(); ++g) {
                            for (int q = 0; q < obs_net_ml.layer_count(); ++q) {
                                delta_mat[g][q] = 0;
                            }
                        }
                        inner_prod = 0;

                        for (int l = 0;l < obs_net_ml.layer_count();++l) {
                            compute_change_stats(i, j, change_stats, k, l);  /// This is change statistics for edge x_{i,j}^(k)                            
                            inner_prod += theta_mat[k][l] * change_stats[k][l];
                            delta_mat[k][l] = change_stats[k][l];
                            if (l != k && (!obs_net_ml.is_edge(i, j, l))) flag = false;
                        }
                        if (flag) cs3 = 1;
                        else cs3 = 0;
                        inner_prod += cs3 * theta[0][6];
                        exp_val = exp(inner_prod);

                        if (obs_net_ml.is_edge(i, j, k)) { // obs_net.get_edge_type(i, j))) { 
                            obs_val = 1;
                        }
                        else {
                            obs_val = 0;
                        }


                      

                        scale_val = (exp_val / (1 + exp_val)) - obs_val;
                        int pp;
                        for (int q = 0; q < obs_net_ml.layer_count();++q) {
                            if (q < k) {
                                pp = 0;
                                for (int g = 0; g < obs_net_ml.layer_count(); ++g) {
                                    for (int f = g; f < obs_net_ml.layer_count(); ++f) {
                                        if (g == q && f == k) delta[pp] = delta_mat[k][q];
                                        pp++;
                                    }
                                }
                            }
                            if (k <= q && k >= 1) {
                                delta[(2 * obs_net_ml.layer_count() - k + 1) * k / 2 + q - k] = delta_mat[k][q];

                            }
                            if (k <= q && k == 0) {
                                delta[q] = delta_mat[k][q];
                            }

                        }
                        delta[6] = cs3;
                        for (int d = 0; d < get_model_dim(); ++d) {
                            gradient[d] += delta[d] * scale_val;
                            //Rcpp::Rcout << "\n  gradient is " << gradient[d] << ", " << "\n";
                        }



                    }

                }
            }

            // Compute gamma scaling parameter and step increment
            proposed_step_sum = 0;
            for (int p = 0; p < get_model_dim(); ++p) {
                proposed_step_sum += pow(gradient[p], 2);
            }
            gamma = 1 / (pow(obs_net_ml.node_count(), 2));


            for (int p = 0; p < get_model_dim(); ++p) {
                theta[1][p] = theta[0][p];  // theta[1] keeps values of last step.
                theta[0][p] -= gamma * gradient[p];
            }

            ///  Assign 1-d vector theta to 2-d vector theta_mat for next updates
            int pp = 0;
            for (int k = 0; k < obs_net_ml.layer_count(); ++k) {
                for (int l = k; l < obs_net_ml.layer_count(); ++l) {
                    theta_mat[k][l] = theta[0][pp];
                    theta_mat[l][k] = theta_mat[k][l];
                    ++pp;
                }
            }

            if (sqrt(proposed_step_sum) < 1e-4 * get_model_dim()) {
                Rcpp::Rcout << "\n    Took " << iter_num << " iterations to converge to initial theta = (";
                for (int p = 0; p < get_model_dim(); ++p) {
                    Rcpp::Rcout << theta[0][p];
                    if (p == (get_model_dim() - 1)) {
                        Rcpp::Rcout << ").\n" << flush;
                    }
                    else {
                        Rcpp::Rcout << ", ";
                    }
                }
                conv_flag = true;
                break;
            }
        }
        if (!conv_flag) {
            Rcpp::Rcout << "\n    Initial estimation procedure failed to converge. Starting estimate may be unstable.";
        }
        for (int p = 0; p < get_model_dim(); ++p) {
            theta_est[p] = theta[0][p];
        }
    }
};

class est_ml {

public:
    sim_ml simulation_ml;
    mlnet obs_net_ml;
    vec theta_est;
    vec obs_vec;
    vec obs_vec_hold;
    vector<double> weights;
    mat information_matrix;
    int NR_max_iter;  /// For Newton Raphson method
    int MCMLE_max_iter;
    int model_dim;
    double NR_tol;
    bool check_chull;

public:
    est_ml(int nsamp, int burn, int intv, int mdim, vector<string> mterms,  /// Use corsslayer_est as the cross-layer parameter name for estimation
        int N, int K, double random_seed,
        int NR_max, double NRtol, int MCMLE_max, bool check_ch, double g)
        : simulation_ml(nsamp, burn, intv, mdim, mterms, N, K, random_seed, g),
        obs_net_ml(N, K),
        theta_est(mdim),
        obs_vec(mdim),
        obs_vec_hold(mdim),
        information_matrix(mdim, mdim)
    {
        NR_max_iter = NR_max;
        NR_tol = NRtol;
        MCMLE_max_iter = MCMLE_max;
        model_dim = mdim;
        check_chull = check_ch;
        weights.resize(nsamp);
    }

    int get_model_dim() {
        return simulation_ml.get_model_dim();
    }

    vector<double> get_theta() {
        return conv_to< stdvec >::from(theta_est);
    }

    vector<double> vec_to_stdvec(vec& vec_) {
        return conv_to< stdvec >::from(vec_);
    }

    vec stdvec_to_vec(vector<double>& vec_) {
        return conv_to< vec >::from(vec_);
    }


    void compute_change_stats(int i, int j, vector<vector<double> >& ch_stat, int k, int l) { 
        ch_stat[k][l] = simulation_ml.m.change_stat_funs_ml[0](i, j, obs_net_ml, k, l);  /// Use cs_crosslayer_est for change stats of 2-layer co-ocurrences
    }

    void compute_initial_estimate() {
        double obs_val;

        vector<vector<double> >  change_stats;   /// Change statistics for each edge
        change_stats.resize(obs_net_ml.layer_count());
        for (int i = 0; i < obs_net_ml.layer_count(); ++i) {
            change_stats[i].resize(obs_net_ml.layer_count());
        }

        vector<vector<double> > theta;
        theta.resize(2);
        theta[0].resize(get_model_dim());
        theta[1].resize(get_model_dim());

        vector<double> gradient;
        gradient.resize(get_model_dim());

        // Compute change statistics for each edge
        
        
 
        // Estimate initial theta using initial guess of zero vector 
        for (int p = 0; p < get_model_dim(); ++p) {
            theta[0][p] = 0;
            theta[1][p] = 0;
        }

        vector<vector<double> >  theta_mat;
        theta_mat.resize(obs_net_ml.layer_count());
        for (int i = 0; i < obs_net_ml.layer_count(); ++i) {
            theta_mat[i].resize(obs_net_ml.layer_count());
        }
        

        bool conv_flag = false;
        double exp_val, scale_val, proposed_step_sum, gamma;
        double inner_prod = 0.0;
        vector<double> delta;
        delta.resize(get_model_dim());
        vector<vector<double> > delta_mat;
        delta_mat.resize(obs_net_ml.layer_count());
        for (int i = 0; i < obs_net_ml.layer_count(); ++i) {
            delta_mat[i].resize(obs_net_ml.layer_count());
        }
        for (int iter_num = 0; iter_num < 30000; ++iter_num) { /// Maximum number of iterations for gradient descent algm.
            // Compute gradient for edge
            for (int p = 0; p < get_model_dim(); ++p) {
                gradient[p] = 0;
            }
            for (int i = 0; i < (obs_net_ml.node_count() - 1); ++i) {
                for (int j = (i + 1); j < obs_net_ml.node_count(); ++j) {
                    for (int k = 0;k < obs_net_ml.layer_count();++k) {

                        bool flag_for1 = false;
                        for (int l = 0;l < obs_net_ml.layer_count();++l) {
                            if (l != k && obs_net_ml.is_edge(i, j, l)) {
                                flag_for1 = true;
                                break;
                            }
                        }
                        if (!flag_for1) continue;

                        for (int d = 0; d < get_model_dim(); ++d) {
                            delta[d] = 0;
                        }
                        for (int g = 0; g < obs_net_ml.layer_count(); ++g) {
                            for (int q = 0; q < obs_net_ml.layer_count(); ++q) {
                                delta_mat[g][q] = 0;
                            }
                        }
                        inner_prod = 0;
                        for (int l = 0;l < obs_net_ml.layer_count();++l) {
                            compute_change_stats(i, j, change_stats, k, l);  /// This is change statistics for edge x_{i,j}^(k)                            
                            inner_prod += theta_mat[k][l] * change_stats[k][l];
                            delta_mat[k][l] = change_stats[k][l];
                            //Rcpp::Rcout << "\n  theta, cs and ip are " << theta_mat[k][l] << ", " << change_stats[k][l]  << "\n";
                        }
                        exp_val = exp(inner_prod);
                        
                        if (obs_net_ml.is_edge(i, j, k)) { // obs_net.get_edge_type(i, j))) { 
                            obs_val = 1;
                        }
                        else {
                            obs_val = 0;
                        }

                        scale_val = (exp_val / (1 + exp_val)) - obs_val;
                        int pp;
                        for (int q = 0; q < obs_net_ml.layer_count();++q) {
                            if (q < k) {
                                pp = 0;
                                for (int g = 0; g < obs_net_ml.layer_count(); ++g) {
                                    for (int f = g; f < obs_net_ml.layer_count(); ++f) {
                                        if (g == q && f == k) delta[pp] = delta_mat[k][q];
                                        pp++;
                                    }
                                }
                            }
                            if (k <= q && k >= 1) {
                                delta[(2*obs_net_ml.layer_count() - k + 1)*k/2 + q - k] = delta_mat[k][q];
                                
                            }
                            if (k <= q && k == 0) {
                                delta[q] = delta_mat[k][q];
                            }
                            
                        }

                        for (int d = 0; d < get_model_dim(); ++d) {
                            gradient[d] += delta[d] * scale_val;
                            //Rcpp::Rcout << "\n  gradient is " << gradient[d] << ", " << "\n";
                        }
                        
                        
                        
                    }
                    
                }
            }

            // Compute gamma scaling parameter and step increment
            proposed_step_sum = 0;
            for (int p = 0; p < get_model_dim(); ++p) {
                proposed_step_sum += pow(gradient[p], 2);
            }
            gamma = 1 / (pow(obs_net_ml.node_count(), 2));
           

            for (int p = 0; p < get_model_dim(); ++p) {
                theta[1][p] = theta[0][p];  // theta[1] keeps values of last step.
                theta[0][p] -= gamma * gradient[p];
            }

            ///  Assign 1-d vector theta to 2-d vector theta_mat for next updates
            int pp = 0;
            for (int k = 0; k < obs_net_ml.layer_count(); ++k) {
                for (int l = k; l < obs_net_ml.layer_count(); ++l) {
                    theta_mat[k][l] = theta[0][pp];
                    theta_mat[l][k] = theta_mat[k][l];
                    ++pp;
                }
            }

            if (sqrt(proposed_step_sum) < 1e-4 * get_model_dim()) {
                Rcpp::Rcout << "\n    Took " << iter_num << " iterations to converge to initial theta = (";
                for (int p = 0; p < get_model_dim(); ++p) {
                    Rcpp::Rcout << theta[0][p];
                    if (p == (get_model_dim() - 1)) {
                        Rcpp::Rcout << ").\n" << flush;
                    }
                    else {
                        Rcpp::Rcout << ", ";
                    }
                }
                conv_flag = true;
                break;
            }
        }
        if (!conv_flag) {
            Rcpp::Rcout << "\n    Initial estimation procedure failed to converge. Starting estimate may be unstable.";
        }
        for (int p = 0; p < get_model_dim(); ++p) {
            theta_est[p] = theta[0][p];
        }
    }

};







class est {

public:
    sim simulation;
    net_overlapping_blocks obs_net;
    vec theta_est;
    vec obs_vec;
    vec obs_vec_hold;
    vector<double> weights;
    mat information_matrix;
    int NR_max_iter;
    int MCMLE_max_iter;
    int model_dim;
    double NR_tol;
    bool check_chull;

public:
    est(int nsamp, int burn, int intv, int mdim, vector<string> mterms,
        int N, vector<vector<int> > bmemb, int K,
        int NR_max, double NRtol, int MCMLE_max, bool check_ch)
        : simulation(nsamp, burn, intv, mdim, mterms, N, bmemb, K),
        obs_net(N, bmemb, K),
        theta_est(mdim),
        obs_vec(mdim),
        obs_vec_hold(mdim),
        information_matrix(mdim, mdim)
    {
        NR_max_iter = NR_max;
        NR_tol = NRtol;
        MCMLE_max_iter = MCMLE_max;
        model_dim = mdim;
        check_chull = check_ch;
        weights.resize(nsamp);
    }

    int get_model_dim() {
        return simulation.get_model_dim();
    }

    vector<double> get_theta() {
        return conv_to< stdvec >::from(theta_est);
    }

    vector<double> vec_to_stdvec(vec& vec_) {
        return conv_to< stdvec >::from(vec_);
    }

    vec stdvec_to_vec(vector<double>& vec_) {
        return conv_to< vec >::from(vec_);
    }

    void compute_obs_stats() {
        for (int p = 0; p < get_model_dim(); ++p) {
            obs_vec.at(p) = simulation.m.stat_funs[p](obs_net, simulation.m.block_restriction[p]);
        }
    }

    void compute_change_stats(int i, int j, vector<double>& ch_stat) {
        int block_restriction;
        int type;
        for (int p = 0; p < get_model_dim(); ++p) {
            block_restriction = simulation.m.block_restriction[p];
            type = obs_net.get_edge_type(i, j);
            ch_stat[p] = simulation.m.change_stat_funs[p](i, j, obs_net, type, block_restriction);
        }
    }

    void compute_initial_estimate() {
        double obs_val;

        vector<vector<vector<double> > > change_stats;
        change_stats.resize(obs_net.node_count());
        for (int i = 0; i < obs_net.node_count(); ++i) {
            change_stats[i].resize(obs_net.node_count());
            for (int j = 0; j < obs_net.node_count(); ++j) {
                change_stats[i][j].resize(get_model_dim());
            }
        }

        vector<vector<double> > theta;
        theta.resize(2);
        theta[0].resize(get_model_dim());
        theta[1].resize(get_model_dim());

        vector<double> gradient;
        gradient.resize(get_model_dim());

        // Compute change statistics for each edge
        for (int i = 0; i < (obs_net.node_count() - 1); ++i) {
            for (int j = (i + 1); j < obs_net.node_count(); ++j) {
                compute_change_stats(i, j, change_stats[i][j]);
                change_stats[j][i] = change_stats[i][j];  /// undirected
            }
        }

        // Estimate initial theta using initial guess of zero vector 
        for (int p = 0; p < get_model_dim(); ++p) {
            theta[0][p] = 0;
            theta[1][p] = 0;
        }

        bool conv_flag = false;
        double exp_val, scale_val, proposed_step_sum, gamma;
        double inner_prod = 0.0;
        for (int iter_num = 0; iter_num < 30000; ++iter_num) {

            // Compute gradient 
            for (int p = 0; p < get_model_dim(); ++p) {
                gradient[p] = 0;
            }
            for (int i = 0; i < (obs_net.node_count() - 1); ++i) {
                for (int j = (i + 1); j < obs_net.node_count(); ++j) {
                    inner_prod = 0;
                    for (int p = 0; p < get_model_dim(); ++p) {
                        inner_prod += theta[0][p] * change_stats[i][j][p];
                    }
                    exp_val = exp(inner_prod);
                    if (obs_net.is_edge(i, j, -1)) { // obs_net.get_edge_type(i, j))) { 
                        obs_val = 1;
                    }
                    else {
                        obs_val = 0;
                    }
                    scale_val = (exp_val / (1 + exp_val)) - obs_val;
                    for (int p = 0; p < get_model_dim(); ++p) {
                        gradient[p] += (change_stats[i][j][p] * scale_val);
                    }
                }
            }

            // Compute gamma scaling parameter and step increment
            proposed_step_sum = 0;
            for (int p = 0; p < get_model_dim(); ++p) {
                proposed_step_sum += pow(gradient[p], 2);
            }
            gamma = 1 / (pow(obs_net.node_count(), 2));

            for (int p = 0; p < get_model_dim(); ++p) {
                theta[1][p] = theta[0][p];
                theta[0][p] -= gamma * gradient[p];
            }

            if (sqrt(proposed_step_sum) < 1e-4 * get_model_dim()) {
                Rcpp::Rcout << "\n    Took " << iter_num << " iterations to converge to initial theta = (";
                for (int p = 0; p < get_model_dim(); ++p) {
                    Rcpp::Rcout << theta[0][p];
                    if (p == (get_model_dim() - 1)) {
                        Rcpp::Rcout << ").\n" << flush;
                    }
                    else {
                        Rcpp::Rcout << ", ";
                    }
                }
                conv_flag = true;
                break;
            }
        }
        if (!conv_flag) {
            Rcpp::Rcout << "\n    Initial estimation procedure failed to converge. Starting estimate may be unstable.";
        }
        for (int p = 0; p < get_model_dim(); ++p) {
            theta_est[p] = theta[0][p];
        }
    }

    void compute_mcmle() {
        int in_ch_count = 0;
        double gamma = 0;
        vec mean(get_model_dim());
        arma::rowvec std_vec(get_model_dim());
        compute_obs_stats();
        obs_vec_hold = obs_vec;
        Rcpp::Rcout << "\n    Observed statistic = (";
        for (int p = 0; p < get_model_dim(); ++p) {
            Rcpp::Rcout << obs_vec.at(p, 0);
            if (p == (get_model_dim() - 1)) {
                Rcpp::Rcout << ").\n" << flush;
            }
            else {
                Rcpp::Rcout << ", ";
            }
        }
        //cout << "\nmade it to estimation check point 4.1\n";

        // MCMLE Iterations: 
        for (int mcmle_iter = 0; mcmle_iter < MCMLE_max_iter; ++mcmle_iter) {
            Rcpp::Rcout << "\n    MCMC-iter " << (mcmle_iter + 1) << ":";

            // Draw samples for MCMC approximation 
            simulation.set_theta(get_theta());
            Rcpp::Rcout << "\n      Simulating with theta = (";
            for (int p = 0; p < get_model_dim(); ++p) {
                Rcpp::Rcout << simulation.theta.at(p);
                if (p == (get_model_dim() - 1)) {
                    Rcpp::Rcout << ").\n" << flush;
                }
                else {
                    Rcpp::Rcout << ", ";
                }
            }
            //cout << "\nmade it to estimation check point 4.2\n";

            simulation.simulate();
            simulation.compute_sample_mean();
            for (int p = 0; p < get_model_dim(); ++p) {
                mean.at(p) = simulation.mean[p];
            }
            Rcpp::Rcout << "\n      Sample mean = (";
            for (int p = 0; p < get_model_dim(); ++p) {
                Rcpp::Rcout << mean.at(p);
                if (p == (get_model_dim() - 1)) {
                    Rcpp::Rcout << ")." << flush;
                }
                else {
                    Rcpp::Rcout << ", ";
                }
            }
            Rcpp::Rcout << "\n      Observed statistic = (";
            for (int p = 0; p < get_model_dim(); ++p) {
                Rcpp::Rcout << obs_vec.at(p, 0);
                if (p == (get_model_dim() - 1)) {
                    Rcpp::Rcout << ").\n" << flush;
                }
                else {
                    Rcpp::Rcout << ", ";
                }
            }
            //cout << "\nmade it to estimation check point 4.3\n";

            // Check if observation is in the convex hull of the simualted sufficient statistics
            if (check_chull) {
                obs_vec = obs_vec_hold;
                bool pass = true;
                //cout << "\nmade it to estimation check point 4.3.0\n";

                if (!is_in_ch(obs_vec, simulation.samples) && !pass) {
                    Rcpp::Rcout << "\n      Observation does not lie within the convex hull of the simulated statistics." << flush;
                    Rcpp::Rcout << "\n      Stepping to convex hull.";
                    std_vec = arma::stddev(simulation.samples, 0);
                    Rcpp::Rcout << "\n\n    Standard deviation vector = (";

                    for (int p = 0; p < get_model_dim(); ++p) {
                        Rcpp::Rcout << std_vec.at(p);
                        if (p == (get_model_dim() - 1)) {
                            Rcpp::Rcout << ")." << flush;
                        }
                        else {
                            Rcpp::Rcout << ", ";
                        }
                    }
                    for (int tt = 1; tt < 201; ++tt) {
                        gamma = (200 - (double)tt) / 200;
                        obs_vec = mean * (1 - gamma) + obs_vec * gamma;
                        if (is_in_ch(obs_vec, simulation.samples)) {
                            if (tt < 200) {
                                gamma = (200 - (double)(tt + 1)) / 200;
                            }
                            Rcpp::Rcout << "\n      Using gamma = " << gamma;
                            obs_vec = mean * (1 - gamma) + obs_vec * gamma;
                            Rcpp::Rcout << "\n      Pseudo-observation is = (";
                            for (int p = 0; p < get_model_dim(); ++p) {
                                Rcpp::Rcout << obs_vec.at(p);
                                if (p == (get_model_dim() - 1)) {
                                    Rcpp::Rcout << ")." << flush;
                                }
                                else {
                                    Rcpp::Rcout << ", ";
                                }
                            }
                            Rcpp::Rcout << "\n\n" << flush;
                            break;
                        }
                    }
                }
                else {
                    in_ch_count += 1;
                    Rcpp::Rcout << "\n      Observation is within the convex hull of the simulated statistics.";
                    Rcpp::Rcout << " Continuing to compute MCMLE." << flush;
                }

            }

            // Do Newton-Raphson estimation with MCMC approximation
            NR_mcmle_optimization();
            Rcpp::Rcout << "\n" << flush;

            if (in_ch_count == 2) break;
        }
        simulation.set_theta(get_theta());
    }


    void NR_mcmle_optimization() {
        int iter = 0;
        double NR_step_len;
        bool conv_flag = false;
        vec step = zeros(get_model_dim());
        vec weighted_sum = zeros(get_model_dim());

        for (int NR_iter = 0; NR_iter < NR_max_iter; ++NR_iter) {
            step.zeros();
            weighted_sum.zeros();
            information_matrix.zeros();

            if (weights.size() != simulation.get_num_samples()) {
                weights.resize(simulation.get_num_samples());
            }

            compute_mcmc_weights();
            compute_information_matrix(weighted_sum);
            compute_theta_step(step, weighted_sum);

            Rcpp::Rcout << "\n\n      Iter " << NR_iter << ":";
            NR_step_len = 1 / (1 + pow(norm(step, 2), 2));
            Rcpp::Rcout << "\n        2-norm of step = " << norm(step, 2);
            Rcpp::Rcout << "\n        step length  = " << NR_step_len;
            theta_est += NR_step_len * step;
            Rcpp::Rcout << "\n        theta = (";
            iter = 0;
            for (double val : theta_est) {
                Rcpp::Rcout << val;
                iter += 1;
                if (iter == theta_est.n_elem) {
                    Rcpp::Rcout << ")." << flush;
                }
                else {
                    Rcpp::Rcout << ", ";
                }
            }

            // Check convergence
            if (norm(step) <= NR_tol) {
                conv_flag = true;
                Rcpp::Rcout << "\n\n            Convergence of NR-proc reached in " << NR_iter << " steps.";
                Rcpp::Rcout << "\n            theta-est = (";
                for (int p = 0; p < get_model_dim(); ++p) {
                    Rcpp::Rcout << theta_est.at(p);
                    if (p == (get_model_dim() - 1)) {
                        Rcpp::Rcout << ").\n" << flush;
                    }
                    else {
                        Rcpp::Rcout << ", ";
                    }
                }
                break;
            }
        }
        if (!conv_flag) {
            Rcpp::Rcout << "\n            NR max iterations hit before convergence. Current estimate may not be accurate." << flush;
        }
    }

    void compute_mcmc_weights() {
        vec theta_diff = zeros(get_model_dim());
        double inner_prod = 0;
        double sum = 0;

        theta_diff = theta_est - stdvec_to_vec(simulation.theta);

        for (int i = 0; i < simulation.get_num_samples(); ++i) {
            inner_prod = dot(simulation.samples.row(i), theta_diff);
            weights[i] = exp(inner_prod);
            sum += weights[i];
        }
        for (int i = 0; i < simulation.get_num_samples(); ++i) {
            weights[i] /= sum;
        }
    }

    void compute_information_matrix(vec& weighted_sum) {
        information_matrix.zeros();
        for (int i = 0; i < simulation.get_num_samples(); ++i) {
            weighted_sum += simulation.samples.row(i).t() * weights.at(i);
            information_matrix += (simulation.samples.row(i).t() * simulation.samples.row(i)) * weights.at(i);
        }
        information_matrix -= (weighted_sum * weighted_sum.t());
    }

    void compute_theta_step(vec& step, vec& weighted_sum) {
        step = solve(information_matrix, obs_vec - weighted_sum);
    }

};