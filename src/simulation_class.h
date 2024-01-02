#pragma once
#include <algorithm>
#include <RcppArmadillo.h>
#include <vector>
#include <string>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "graph_class.h"
#include "model.h"

using namespace std;
using namespace arma;




class sim_ml_exact {

public:
    vector<double>  theta; // Allow heterogeneity within and across layers. Should specify the highest order of interactions.
    vector<vector<double> > theta_mat;
    vector<double> chain_state;
    vector<double> mean;
    int num_samples;
    int burnin;
    int interval;
    int model_dim;
    double random_seed;
    mat samples;
    mlnet mlnetwork;
    mod m;
    vector<int> count_exact;
    vector<int> count_net_exact;
    vector<vector<int> >  count_cross_exact;
    vector<vector<int> > netCount_samp_exact;
    vector<vector<vector<int> > > crossLayerCount_samp_exact;
    vector<double> ut;
    double Z1; // normalizing constant for pdf
    int count_threeway;
    vector<int> threewayCount_samp;


public:
    sim_ml_exact(int nsamp, int burn, int intv, int mdim, vector<string> mterms,
        int N, int K, double rand_seed)
        : samples(nsamp, mdim),
        mlnetwork(N, K),
        m(mterms)
    {
        num_samples = nsamp;
        burnin = burn;
        interval = intv;
        model_dim = mdim;
        theta.resize(get_model_dim());
        random_seed = rand_seed;
        ut.resize(2^mlnetwork.layer_count());
    }

    void simulate_ml_exact() {

        double u;
        int samp_num = 0;

        std::random_device rd;
        std::mt19937 g(rd());

        std::random_device rand_dev;
        //Rcpp::Rcout << "The random device is: " << rand_dev() << "\n";
        std::mt19937 rand(random_seed);
        //Rcpp::Rcout << "the random seed is: " << rand << "\n";
        std::uniform_real_distribution<> runif(0.0, 1.0);

        //initial_samp(); /// Sample the network with prob = 0.5 for each edge independently as an initial network
        //Run burnin
        double p1 = ut[0] / Z1;
        double p2 = (ut[0] + ut[1]) / Z1;
        double p3 = (ut[0] + ut[1] + ut[2]) / Z1;
        double p4 = (ut[0] + ut[1] + ut[2] + ut[3]) / Z1;
        double p5 = (ut[0] + ut[1] + ut[2] + ut[3] + ut[4]) / Z1;
        double p6 = (ut[0] + ut[1] + ut[2] + ut[3] + ut[4] + ut[5]) / Z1;
        double p7 = (ut[0] + ut[1] + ut[2] + ut[3] + ut[4] + ut[5] + ut[6]) / Z1;

        
        count_net_exact.resize(mlnetwork.layer_count());
        count_cross_exact.resize(mlnetwork.layer_count());
        for (int i = 0;i < mlnetwork.layer_count();++i) {
            count_cross_exact[i].resize(mlnetwork.layer_count());
        }
        netCount_samp_exact.resize(mlnetwork.layer_count());
        Rcpp::Rcout << netCount_samp_exact.size() << "\n";
        crossLayerCount_samp_exact.resize(mlnetwork.layer_count());
        threewayCount_samp.resize(num_samples);


        Rcpp::Rcout << "sim_ml check point 1.4\n";


        for (int i = 0;i < mlnetwork.layer_count();++i) {
            netCount_samp_exact[i].resize(num_samples);
            Rcpp::Rcout << "one step before assignment\n";
            netCount_samp_exact[i][num_samples - 1] = 100;
            Rcpp::Rcout << netCount_samp_exact[i].size() << "\n";
            Rcpp::Rcout << i << ", " << netCount_samp_exact[i][num_samples-1] << "\n";
        }
        Rcpp::Rcout << "sim_ml check point 1.5\n";
        for (int i = 0;i < mlnetwork.layer_count();++i) {
            Rcpp::Rcout << "sim_ml check point 1.6\n";
            crossLayerCount_samp_exact[i].resize(mlnetwork.layer_count());
            Rcpp::Rcout << crossLayerCount_samp_exact[i].size() << "\n";
            Rcpp::Rcout << "sim_ml check point 1.7\n";
            for (int j = 0;j < mlnetwork.layer_count();++j) {
                Rcpp::Rcout << "sim_ml check point 1.8\n";
                crossLayerCount_samp_exact[i][j].resize(num_samples);
                Rcpp::Rcout << crossLayerCount_samp_exact[i][j].size() << "\n";
                Rcpp::Rcout << "sim_ml check point 1.9\n";
                cout << i << ", " << j << ", " << crossLayerCount_samp_exact[i][j][num_samples-1] << "\n";
            }
        }
        Rcpp::Rcout << "sim_ml check point 2\n";

        int s = 0;
        while (samp_num < num_samples) {
            
            count_net_exact.assign(mlnetwork.layer_count(), 0);
            for (int i = 0;i < mlnetwork.layer_count();++i) {
                count_cross_exact[i].assign(mlnetwork.layer_count(),0);
                
            }
            cout << "sim_ml check point 3\n";


            for (int i = 0;i < mlnetwork.node_count();++i) {
                for (int j = i + 1;j < mlnetwork.node_count();++j) {
                    
                    u = runif(rand);
                    if (u >= p1 && u < p2) {
                        mlnetwork.add_edge(i, j, 2);
                    }
                    if (u >= p2 && u < p3) {
                        mlnetwork.add_edge(i, j, 1);
                    }
                    if (u >= p3 && u < p4 ) {
                        mlnetwork.add_edge(i, j, 1);
                        mlnetwork.add_edge(i, j, 2);
                    }
                    if (u >= p4 && u < p5) {
                        mlnetwork.add_edge(i, j, 0);
                    }
                    if (u >= p5 && u < p6) {
                        mlnetwork.add_edge(i, j, 0);
                        mlnetwork.add_edge(i, j, 2);
                    }
                    if (u >= p6 && u < p7) {
                        mlnetwork.add_edge(i, j, 0);
                        mlnetwork.add_edge(i, j, 1);
                    }
                    if (u >= p7 && u < 1) {
                        mlnetwork.add_edge(i, j, 0);
                        mlnetwork.add_edge(i, j, 1);
                        mlnetwork.add_edge(i, j, 2);
                    }

                    //count the number of edges in each layer                    
                    for (auto ele : mlnetwork.adj[i][j]) {
                        count_net_exact[ele] += 1;
                    }




                    //count the number of co-occurences between 2 layers

                    for (auto ele : mlnetwork.adj[i][j]) {
                        for (auto ele2 : mlnetwork.adj[i][j]) {
                            if (ele >= ele2) continue;
                            count_cross_exact[ele][ele2] += 1;
                        }

                    }

                    if (mlnetwork.adj[i][j].size() == 3) count_threeway += 1;
                }
            }
            cout << "sim_ml check point 4\n";

            for (int i = 0;i < mlnetwork.layer_count();++i) {
                netCount_samp_exact[i][samp_num] = count_net_exact[i];

            }


            for (int i = 0;i < mlnetwork.layer_count() - 1;++i) {
                for (int j = i + 1;j < mlnetwork.layer_count();++j) {
                    crossLayerCount_samp_exact[i][j][samp_num] = count_cross_exact[i][j];
                }
            }

            threewayCount_samp[samp_num] = count_threeway;
            

            //record_sample(samp_num);
            samp_num += 1;

            for (int e1 = 0; e1 < mlnetwork.node_count();++e1) {
                for (int e2 = e1+1; e2 < mlnetwork.node_count();++e2) {
                    mlnetwork.adj[e1][e2].clear();
                    mlnetwork.adj[e2][e1].clear();
                }
            }

        }



    }




    int get_model_dim() {
        return model_dim;
    }


    void set_theta(vector<double> vec) {
        theta = vec;
        vector<double> p;
        p.resize(get_model_dim());

        p[0] = exp(theta[0]) / (1 + exp(theta[0]));
        p[3] = exp(theta[3]) / (1 + exp(theta[3]));
        p[5] = exp(theta[5]) / (1 + exp(theta[5]));

        p[1] = exp(theta[1]);
        p[2] = exp(theta[2]);
        p[4] = exp(theta[4]);
        p[6] = exp(theta[6]);

        ut[0] = (1 - p[0]) * (1 - p[3]) * (1 - p[5]);
        ut[1] = (1 - p[0]) * (1 - p[3]) * p[5];
        ut[2] = (1 - p[0]) * p[3] * (1 - p[5]);
        ut[3] = (1 - p[0]) * p[3] * p[5] * p[4];
        ut[4] = p[0] * (1 - p[3]) * (1 - p[5]);
        ut[5] = p[0] * (1 - p[3]) * p[5] * p[2];
        ut[6] = p[0] * p[3] * (1 - p[5]) * p[1];
        ut[7] = p[0] * p[1] * p[2] * p[3] * p[4] * p[5] * p[6];
        Z1 = ut[0] + ut[1] + ut[2] + ut[3] + ut[4] + ut[5] + ut[6] + ut[7];
    }



    // Returns factorial of n
    int fact(int n)
    {
        int res = 1;
        for (int i = 2; i <= n; i++)
            res = res * i;
        return res;
    }


    int nCr(int n, int r)
    {
        return fact(n) / (fact(r) * fact(n - r));
    }


};




class sim_ml_Hway {

public:
    vector<double>  theta; // Allow heterogeneity within and across layers. Should specify the highest order of interactions.
    vector<double> mean;
    int num_samples;
    int burnin;
    int interval;
    int model_dim;
    double random_seed;
    mat samples;
    mlnet mlnetwork;
    mod m;
    vector<int> count;
    vector<int> count_net;
    int count_threeway;
    vector<vector<vector<int> > > count_cross;
    vector<vector<int> > netCount_samp;
    vector<int> threewayCount_samp;
    vector<vector<vector<int> > > crossLayerCount_samp;
    double gy;  //specify density g(y) of y
    int H;
    vector<int> all_interaction_layer;
    vector<int> combination;
    vector<vector <int> > selected_layer;


public:
    sim_ml_Hway(int nsamp, int burn, int intv, int mdim, vector<string> mterms,
        int N, int K, int highest_order, double rand_seed, double g)
        : samples(nsamp, mdim),
        mlnetwork(N, K),
        m(mterms)
    {
        num_samples = nsamp;
        burnin = burn;
        interval = intv;
        model_dim = mdim;
        theta.resize(get_model_dim());
        mean.resize(get_model_dim());
        random_seed = rand_seed;
        gy = g;
        H = highest_order;

    }

    int compute_change_stats(int i, int j, int k, vector <int>& layers) {
        bool f = true;
        for (int ii = 0; ii < layers.size(); ii++) {
            if (!mlnetwork.is_edge(i, j, layers[ii]) && layers[ii] != k) {
                f = false;
                break;
            }
        }
        if (f) return 1;
        else return 0;
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

    

    void simulate_ml() {
        vector<double>  change_stat;   /// Change statistics for each edge
        change_stat.resize(get_model_dim());   //need to be changed to the number of dimensions);
        for (int e = 0; e < mlnetwork.layer_count(); ++e) {
            all_interaction_layer.push_back(e);
        }

        double inner_prod = 0.0;
        int max_dim = 0;
        int para_dim = 0;
        int s = 0;
        double prop_prob, u;
        int samp_num = 0;

        std::random_device rd;
        std::mt19937 g(rd());

        std::random_device rand_dev;
        //Rcpp::Rcout << "The random device is: " << rand_dev() << "\n";
        std::mt19937 rand(random_seed);
        //Rcpp::Rcout << "the random seed is: " << rand << "\n";
        std::uniform_real_distribution<> runif(0.0, 1.0);

        initial_samp(); /// Sample the network with prob = 0.5 for each edge independently as an initial network
        //cout << "sim_ml check point 1\n";
        //Run burnin
        bool flag;
        for (int i = 0;i < mlnetwork.node_count();++i) {
            for (int j = i + 1;j < mlnetwork.node_count();++j) {
                if (runif(rand) < gy) {
                    for (int upd_num = 0; upd_num < burnin; ++upd_num) {

                        for (int k = 0;k < mlnetwork.layer_count();++k) {
                            for (int p = 0; p < get_model_dim();++p) {
                                change_stat[p] = 0;
                            }

                            inner_prod = 0.0;
                            max_dim = 0;
                            para_dim = 0;
                            for (int h = 1; h <= H; ++h) {
                                select_layer(0, h);
                                for (auto ele : selected_layer) {
                                    if (binary_search(ele.begin(), ele.end(), k)) {
                                        change_stat[para_dim] = compute_change_stats(i, j, k, ele);
                                        inner_prod += theta[para_dim] * change_stat[para_dim];
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

                            prop_prob = 1 / (1 + exp(inner_prod)); 

                            //Rcpp::Rcout << "second prob: " << prop_prob << "\n";


                            u = runif(rand);
                            if (u <= prop_prob) {
                                if (mlnetwork.is_edge(i, j, k)) {
                                    mlnetwork.delete_edge(i, j, k);
                                   
                                }
                            }
                            else {
                                if (!mlnetwork.is_edge(i, j, k)) {
                                    mlnetwork.add_edge(i, j, k);
                                }
                            }


                            // Enforce h(x,y) = 1, use g(y = 1) = 0.333
                            flag = false;
                            for (int kk = 0; kk < mlnetwork.layer_count(); ++kk) {
                                if (mlnetwork.is_edge(i, j, kk)) {
                                    flag = true;
                                    break;
                                }
                            }

                            if (!flag) {
                                mlnetwork.add_edge(i, j, k);
                            }



                        }
                    }
                }
                else {
                    for (int kk = 0; kk < mlnetwork.layer_count(); ++kk) {
                        if (mlnetwork.is_edge(i, j, kk)) {
                            mlnetwork.delete_edge(i, j, kk);
                        }
                    }

                }
            }
        }
        //cout << "sim_ml check point 1.3\n";

        //select dyad for statistics tracking
        int loc1 = 0;
        int loc2 = 0;
        double r1;
        double r2;
        int M;
        int m1;
        int m2;
    
        count_cross.resize(mlnetwork.layer_count());
        netCount_samp.resize(mlnetwork.layer_count());
        crossLayerCount_samp.resize(mlnetwork.layer_count());
        threewayCount_samp.resize(num_samples);


        r1 = runif(rand);
        r2 = runif(rand);
      


        for (int i = 0;i < mlnetwork.layer_count();++i) {
            netCount_samp[i].resize(num_samples);
        }

        for (int i = 0;i < mlnetwork.layer_count();++i) {
            crossLayerCount_samp[i].resize(mlnetwork.layer_count());
            for (int j = 0;j < mlnetwork.layer_count();++j) {
                crossLayerCount_samp[i][j].resize(num_samples);
            }
        }

        s = 0;
        while (samp_num < num_samples) {

            count_net.assign(mlnetwork.layer_count(), 0);
            count_threeway = 0;
            for (int i = 0;i < mlnetwork.layer_count();++i) {
                count_cross[i].resize(mlnetwork.layer_count());
                for (int j = 0;j < mlnetwork.layer_count();++j) {
                    count_cross[i][j].assign(1, 0);
                }
            }

            for (int i = 0;i < mlnetwork.node_count();++i) {
                for (int j = i + 1;j < mlnetwork.node_count();++j) {
                    if (runif(rand) < gy) {
                        for (int upd_num = 0; upd_num < interval; ++upd_num) {

                            for (int k = 0;k < mlnetwork.layer_count();++k) {
                                for (int p = 0; p < get_model_dim();++p) {
                                    change_stat[p] = 0;
                                }

                                inner_prod = 0.0;
                                max_dim = 0;
                                para_dim = 0;
                                for (int h = 1; h <= H; ++h) {
                                    select_layer(0, h);
                                    for (auto ele : selected_layer) {
                                        if (binary_search(ele.begin(), ele.end(), k)) {
                                            change_stat[para_dim] = compute_change_stats(i, j, k, ele);
                                            inner_prod += theta[para_dim] * change_stat[para_dim];
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

                                prop_prob = 1 / (1 + exp(inner_prod));

                                //Rcpp::Rcout << "second prob: " << prop_prob << "\n";


                                u = runif(rand);
                                if (u <= prop_prob) {
                                    if (mlnetwork.is_edge(i, j, k)) {
                                        mlnetwork.delete_edge(i, j, k);

                                    }
                                }
                                else {
                                    if (!mlnetwork.is_edge(i, j, k)) {
                                        mlnetwork.add_edge(i, j, k);
                                    }
                                }


                                // Enforce h(x,y) = 1, use g(y = 1) = 0.333
                                flag = false;
                                for (int kk = 0; kk < mlnetwork.layer_count(); ++kk) {
                                    if (mlnetwork.is_edge(i, j, kk)) {
                                        flag = true;
                                        break;
                                    }
                                }

                                if (!flag) {
                                    mlnetwork.add_edge(i, j, k);
                                }



                            }
                        }
                    }
                    else {
                        for (int kk = 0; kk < mlnetwork.layer_count(); ++kk) {
                            if (mlnetwork.is_edge(i, j, kk)) {
                                mlnetwork.delete_edge(i, j, kk);
                            }
                        }

                    }
                }
            }

            for (int i = 0;i < mlnetwork.node_count();++i) {
                for (int j = i + 1;j < mlnetwork.node_count();++j) {
                    //count the number of edges in each layer                    
                    for (auto ele : mlnetwork.adj[i][j]) {
                        count_net[ele] += 1;
                    }




                    //count the number of co-occurences between 2 layers

                    for (auto ele : mlnetwork.adj[i][j]) {
                        for (auto ele2 : mlnetwork.adj[i][j]) {
                            if (ele >= ele2) continue;
                            m1 = min(ele, ele2);
                            m2 = max(ele, ele2);
                            count_cross[m1][m2][0] += 1;
                        }

                    }

                    // count the 3-way interactions
                    if (mlnetwork.adj[i][j].size() >= 3) count_threeway += 1;




                }
            }

            for (int i = 0;i < mlnetwork.layer_count();++i) {
                netCount_samp[i][samp_num] = count_net[i];

            }

            for (int i = 0;i < mlnetwork.layer_count() - 1;++i) {
                for (int j = i + 1;j < mlnetwork.layer_count();++j) {
                    crossLayerCount_samp[i][j][samp_num] = count_cross[i][j][0];
                }
            }

            threewayCount_samp[samp_num] = count_threeway;
            samp_num += 1;







            

        }



    }

    
    ///uniformly initialize the sampler
    void initial_samp() {
        std::random_device rd;
        std::mt19937 g(rd());

        std::random_device rand_dev;
        std::mt19937 rand(random_seed + 1); /// make this seed different from the seed for 
        //Rcpp::Rcout << "the random seed in initial sampling is: " << rand << "\n";
        std::uniform_real_distribution<> runif(0.0, 1.0);
        for (int i = 0;i < mlnetwork.node_count();++i) {
            for (int j = i + 1;j < mlnetwork.node_count();++j) {
                for (int k = 0;k < mlnetwork.layer_count();++k) {
                    if (runif(rand) < 0.5) mlnetwork.add_edge(i, j, k);
                }
            }
        }
    }

    



    int get_model_dim() {
        return model_dim;
    }

    
    void set_theta(vector<double> vec) {
        theta = vec;
        
    }

    void compute_sample_mean() {
        rowvec col_mean(get_model_dim());
        col_mean = arma::mean(samples, 0);
        for (int p = 0; p < mean.size(); ++p) {
            mean[p] = col_mean.at(p);
        }
    }

    


    // Returns factorial of n
    int fact(int n)
    {
        int res = 1;
        for (int i = 2; i <= n; i++)
            res = res * i;
        return res;
    }


    int nCr(int n, int r)
    {
        return fact(n) / (fact(r) * fact(n - r));
    }


};



class sim_ml {

public:
    vector<double>  theta; // Vector form of theta. Allow heterogeneity within and across layers. Should specify the highest order of interactions.
    vector<vector<double> > theta_mat; // Matrix form of theta
    vector<double> chain_state;
    vector<double> mean;
    int num_samples;
    int burnin;
    int interval;
    int model_dim;
    int cs3; // change statistic for 3-way interactions (when K = 3).
    double random_seed;
    mat samples;
    mlnet mlnetwork;
    mod m;
    vector<int> count;
    vector<int> count_net;
    int count_threeway;
    vector<vector<vector<int> > > count_cross;
    vector<vector<int> > netCount_samp;
    vector<int> threewayCount_samp;
    vector<vector<vector<int> > > crossLayerCount_samp;
    double gy;  //specify density g(y) of y (when Y is assumed to be a Bernoulli graph)


public:
    sim_ml(int nsamp, int burn, int intv, int mdim, vector<string> mterms,
        int N, int K, double rand_seed, double g)
        : samples(nsamp, mdim),
        mlnetwork(N, K),
        m(mterms)
    {
        num_samples = nsamp;
        burnin = burn;
        interval = intv;
        model_dim = mdim;
        chain_state.resize(get_model_dim());
        theta.resize(get_model_dim());
        mean.resize(get_model_dim());
        random_seed = rand_seed;
        theta_mat.resize(mlnetwork.layer_count());
        for (int i = 0;i < mlnetwork.layer_count();++i) {
            theta_mat[i].resize(mlnetwork.layer_count());
        }
        gy = g;
        
    }

    void simulate_ml() {
        vector<vector<double> > change_stat;
        change_stat.resize(mlnetwork.layer_count());
        for (int i = 0;i < mlnetwork.layer_count();++i) {
            change_stat[i].resize(mlnetwork.layer_count());
        }

        
        
        double prop_prob, u;
        int samp_num = 0;

        std::random_device rd;
        std::mt19937 g(rd());

        std::random_device rand_dev;
        //Rcpp::Rcout << "The random device is: " << rand_dev() << "\n";
        std::mt19937 rand(random_seed);
        //Rcpp::Rcout << "the random seed is: " << rand << "\n";
        std::uniform_real_distribution<> runif(0.0, 1.0);

        initial_samp(); /// Sample the network with prob = 0.5 for each edge independently as an initial network
        //Run burnin
        bool flag;
        for (int i = 0;i < mlnetwork.node_count();++i) {
            for (int j = i+1;j < mlnetwork.node_count();++j) {
                if (runif(rand) < gy) {
                    for (int upd_num = 0; upd_num < burnin; ++upd_num) {
                        for (int k = 0;k < mlnetwork.layer_count();++k) {
                            for (int l = 0; l < mlnetwork.layer_count(); ++l) {
                                compute_change_stat(i, j, change_stat, k, l);  /// This is the change statistic for x_{i,j}^(k) given the rest layers, i.e., (k,l) and (l,k) are different.
                                
                            }
                            

                            prop_prob = compute_cond_prob(i, j, k, change_stat);
                            

                            u = runif(rand);
                            if (u <= prop_prob) {
                                if (mlnetwork.is_edge(i, j, k)) {
                                    mlnetwork.delete_edge(i, j, k);
                                    update_chain(-1, change_stat);
                                }
                            }
                            else {
                                if (!mlnetwork.is_edge(i, j, k)) {
                                    mlnetwork.add_edge(i, j, k);
                                    update_chain(1, change_stat);
                                }
                            }


                            // Enforce h(x,y) = 1, use g(y = 1) = 0.333
                            bool h = 0;
                            for (int kk = 0; kk < mlnetwork.layer_count(); ++kk) {
                                if (mlnetwork.is_edge(i, j, kk)) {
                                    h = 1;
                                    break;
                                }
                            }

                            if (h == 0) {
                                mlnetwork.add_edge(i, j, k);
                                update_chain(1, change_stat);
                            }



                        }
                    }
                }
                else {
                    for (int k = 0; k < mlnetwork.layer_count(); ++k) {
                        if (mlnetwork.is_edge(i, j, k)) {
                            mlnetwork.delete_edge(i, j, k);
                            update_chain(-1, change_stat);
                        }
                    }

                }
            }
        }

        //select dyad for statistics tracking
        int loc1 = 0;
        int loc2 = 0;
        double r1;
        double r2;
        int M;
        int m1;
        int m2;
        
        count_cross.resize(mlnetwork.layer_count());
        netCount_samp.resize(mlnetwork.layer_count());
        crossLayerCount_samp.resize(mlnetwork.layer_count());
        threewayCount_samp.resize(num_samples);


        r1 = runif(rand);
        r2 = runif(rand);
        
        

        for (int i = 0;i < mlnetwork.layer_count();++i) {
            netCount_samp[i].resize(num_samples);
        }

        for (int i = 0;i < mlnetwork.layer_count();++i) {
            crossLayerCount_samp[i].resize(mlnetwork.layer_count());
            for (int j = 0;j < mlnetwork.layer_count();++j) {
                crossLayerCount_samp[i][j].resize(num_samples);
            }
        }

        int s = 0;
        while (samp_num < num_samples) {

            count_net.assign(mlnetwork.layer_count(),0);
            count_threeway = 0;
            for (int i = 0;i < mlnetwork.layer_count();++i) {
                count_cross[i].resize(mlnetwork.layer_count());
                for (int j = 0;j < mlnetwork.layer_count();++j) {
                    count_cross[i][j].assign(1,0);
                }
            }

            for (int i = 0;i < mlnetwork.node_count();++i) {
                for (int j = i + 1;j < mlnetwork.node_count();++j) {
                    if (runif(rand) < gy) {
                        for (int upd_num = 0; upd_num < interval; ++upd_num) {  
                            for (int k = 0;k < mlnetwork.layer_count();++k) {
                                //flag = true;
                                for (int l = 0; l < mlnetwork.layer_count();++l) {
                                    compute_change_stat(i, j, change_stat, k, l);  /// This is the change statistic for x_{i,j}^(k) given the rest layers, i.e., (k,l) and (l,k) are different.
                                   
                                }
                               


                                prop_prob = compute_cond_prob(i, j, k, change_stat);
                                

                                u = runif(rand);
                                if (u <= prop_prob) {
                                    if (mlnetwork.is_edge(i, j, k)) {
                                        mlnetwork.delete_edge(i, j, k);
                                        update_chain(-1, change_stat);
                                    }
                                }
                                else {
                                    if (!mlnetwork.is_edge(i, j, k)) {
                                        mlnetwork.add_edge(i, j, k);
                                        update_chain(1, change_stat);
                                    }
                                }


                                // Enforce h(x,y) = 1, use g(y = 1) = 0.333
                                bool h = 0;
                                for (int kk = 0; kk < mlnetwork.layer_count(); ++kk) {
                                    if (mlnetwork.is_edge(i, j, kk)) {
                                        h = 1;
                                        break;
                                    }
                                }

                                if (h == 0) {
                                    mlnetwork.add_edge(i, j, k);
                                    update_chain(1, change_stat);
                                }



                            }
                        }
                    }
                    else {
                        for (int k = 0; k < mlnetwork.layer_count(); ++k) {
                            if (mlnetwork.is_edge(i, j, k)) {
                                mlnetwork.delete_edge(i, j, k);
                                update_chain(-1, change_stat);
                            }
                        }

                    }
                }
            }

            for (int i = 0;i < mlnetwork.node_count();++i) {
                for (int j = i + 1;j < mlnetwork.node_count();++j) {
                    //count the number of edges in each layer                    
                    for (auto ele : mlnetwork.adj[i][j]) {
                        count_net[ele] += 1;
                    }
            
                    
                    

                    //count the number of co-occurences between 2 layers
                    
                    for (auto ele : mlnetwork.adj[i][j]) {
                        for (auto ele2 : mlnetwork.adj[i][j]) {
                            if (ele >= ele2) continue;
                            m1 = min(ele, ele2);
                            m2 = max(ele, ele2);                            
                            count_cross[m1][m2][0] += 1;
                        }
                        
                    }

                    // count the 3-way interactions
                    if (mlnetwork.adj[i][j].size() == 3) count_threeway += 1;
                    

                    

                }
            }
            //cout << "sim_ml check point 4\n";

            for (int i = 0;i < mlnetwork.layer_count();++i) {
                netCount_samp[i][samp_num] = count_net[i];
                
            }

            for (int i = 0;i < mlnetwork.layer_count() - 1;++i) {
                for (int j = i + 1;j < mlnetwork.layer_count();++j) {
                    crossLayerCount_samp[i][j][samp_num] = count_cross[i][j][0];
                }
            }
            
            threewayCount_samp[samp_num] = count_threeway;


            


            record_sample(samp_num);
            samp_num += 1;

        }

        
        
    }

    void record_sample(int ind) {
        for (int p = 0; p < get_model_dim(); ++p) {
            samples.at(ind, p) = chain_state[p];
        }
    }
    ///uniformly initialize the sampler
    void initial_samp() {
        std::random_device rd;
        std::mt19937 g(rd());

        std::random_device rand_dev;
        std::mt19937 rand(random_seed+1); /// make this seed different from the seed for 
        //Rcpp::Rcout << "the random seed in initial sampling is: " << rand << "\n";
        std::uniform_real_distribution<> runif(0.0, 1.0);
        for (int i = 0;i < mlnetwork.node_count();++i) {
            for (int j = i + 1;j < mlnetwork.node_count();++j) {
                for (int k = 0;k < mlnetwork.layer_count();++k) {
                    if (runif(rand) < 0.5) mlnetwork.add_edge(i, j, k);
                }
            }
        }
    }

    /// compute conditional probability for x_{i,j}^(k) = 0 given the rest layers of the same dyad
    double compute_cond_prob(int i, int j, int k, vector<vector<double> >& change_stat) {
        
        double val = 0.0;
        double prob;
        for (int l = 0;l < mlnetwork.layer_count();++l) {
            val += theta_mat[k][l] * change_stat[k][l];
        }

        prob = 1 / (1 + exp(val));  /// Prob that x_{i,j}^(k) = 0.
        return prob;
    }

    

    int get_model_dim() {
        return model_dim;
    }

    void compute_change_stat(int i, int j, vector<vector<double> >& change_stat, int k, int l) {
        change_stat[k][l] = m.change_stat_funs_ml[0](i, j, mlnetwork, k,l);
    }

    void set_theta(vector<double> vec) {
        theta = vec;
        int p = 0;
        for (int i = 0;i < mlnetwork.layer_count();++i) {
            for (int j = i;j < mlnetwork.layer_count();++j) {
                theta_mat[i][j] = theta[p];
                theta_mat[j][i] = theta_mat[i][j];
                ++p;
            }
        }
    }

    void compute_sample_mean() {
        rowvec col_mean(get_model_dim());
        col_mean = arma::mean(samples, 0);
        for (int p = 0; p < mean.size(); ++p) {
            mean[p] = col_mean.at(p);
        }
    }

    void update_chain(int sign, vector<vector<double>>& change_stat) {
        for (int p = 0; p < get_model_dim(); ++p) {
            chain_state[p] += sign * change_stat[int(p/mlnetwork.layer_count())][p%mlnetwork.layer_count()];
        }
    }


    // Returns factorial of n
    int fact(int n)
    {
        int res = 1;
        for (int i = 2; i <= n; i++)
            res = res * i;
        return res;
    }


    int nCr(int n, int r)
    {
        return fact(n) / (fact(r) * fact(n - r));
    }


};

class sim {

public:
    vector<double> theta;
    vector<double> chain_state;
    vector<double> mean;
    int num_samples;
    int burnin;
    int interval;
    int model_dim;
    mat samples;
    net_overlapping_blocks network;
    mod m;

public:
    sim(int nsamp, int burn, int intv, int mdim, vector<string> mterms,
        int N, vector<vector<int> > bmemb, int K)
        : samples(nsamp, mdim),
        network(N, bmemb, K),
        m(mterms)
    {
        num_samples = nsamp;
        burnin = burn;
        interval = intv;
        model_dim = mdim;
        chain_state.resize(get_model_dim());
        theta.resize(get_model_dim());
        mean.resize(get_model_dim());
    }

    int get_model_dim() {
        return model_dim;
    }

    void set_theta(vector<double> vec) {
        theta = vec;
    }

    vector<double> get_theta() {
        return theta;
    }

    int get_num_samples() {
        return num_samples;
    }

    vector<double> get_chain_state() {
        return chain_state;
    }

    void record_sample(int ind) {
        for (int p = 0; p < get_model_dim(); ++p) {
            samples.at(ind, p) = chain_state[p];
        }
    }

    void compute_sample_mean() {
        rowvec col_mean(get_model_dim());
        col_mean = arma::mean(samples, 0);
        for (int p = 0; p < mean.size(); ++p) {
            mean[p] = col_mean.at(p);
        }
    }

    void compute_change_stat(int i, int j, vector<double>& change_stat, int type) {
        for (int p = 0; p < get_model_dim(); ++p) {
            change_stat[p] = m.change_stat_funs[p](i, j, network, type, m.block_restriction[p]);
        }
        
    }

    double compute_prop_prob(int i, int j, vector<double>& change_stat, int type) {
        double val = 0.0;
        double prob;
        for (int t = 0; t < get_model_dim(); ++t) {
            val += theta[t] * change_stat[t];
        }
        if (!network.is_edge(i, j, type)) {
            val *= -1;
        }
        prob = 1 / (1 + exp(val));
        return prob;
    }

    void update_chain(int sign, vector<double>& change_stat) {
        for (int p = 0; p < get_model_dim(); ++p) {
            chain_state[p] += sign * change_stat[p];
        }
    }

    void select_dyad_unique(int k, int l, vector<int>& dyad, double u1, double u2) {
        int loc;
        loc = get_int(u1, network.unique[k].size()); ///randomly select an integer between 0 and network.unique[k].size()-1 (select a node in unique[k] randomly )
        dyad[0] = network.unique[k][loc];
        if (k == l) {
            swap(network.unique[k][loc], network.unique[k].back());
            loc = get_int(u2, network.unique[l].size() - 1);
            dyad[1] = network.unique[l][loc];
        }
        else {
            loc = get_int(u2, network.unique[l].size());
            dyad[1] = network.unique[l][loc];
        }

    }

   
    void select_dyad_overlap(int k, vector<int>& dyad, double u1, double u2) {
        int loc;
        loc = get_int(u1, network.overlap[k].size());
        dyad[0] = network.overlap[k][loc];
        loc = get_int(u2, network.node_neighborhoods[dyad[0]].size());
        dyad[1] = network.node_neighborhoods[dyad[0]][loc];
    }

    
    void select_dyad_between(int k, int l, vector<int>& dyad, double u1, double u2) {
        int loc;
        loc = get_int(u1, network.between[k][l][0].size());
        dyad[0] = network.between[k][l][0][loc];
        loc = get_int(u2, network.between[k][l][1].size());
        dyad[1] = network.between[k][l][1][loc];
    }

    

    void simulate() { ///this code cannot deal with one-block network
        double prop_prob, u, u1, u2;
        int samp_num = 0;
        int type;
        vector<int> dyad;
        dyad.resize(2);
        vector<double> change_stat;
        change_stat.resize(get_model_dim());

        std::random_device rd;
        std::mt19937 g(rd());

        std::random_device rand_dev;
        std::mt19937 rand(rand_dev());
        std::uniform_real_distribution<> runif(0.0, 1.0);
        // ** Run burnin period 
        // First: Unique subgraphs 
        type = 2;
        for (int k = 0; k < network.block_count(); ++k) {
            for (int upd_num = 0; upd_num < burnin; ++upd_num) {
                u1 = runif(rand);
                u2 = runif(rand);
                if (network.unique[k].size() <= 1) break; 
                select_dyad_unique(k, k, dyad, u1, u2);
                compute_change_stat(dyad[0], dyad[1], change_stat, type);
                prop_prob = compute_prop_prob(dyad[0], dyad[1], change_stat, type);
                u = runif(rand);
                if (u <= prop_prob) {
                    if (network.is_edge(dyad[0], dyad[1], type)) {
                        network.delete_edge(dyad[0], dyad[1], type);
                        update_chain(-1, change_stat);
                    }
                    else {
                        network.add_edge(dyad[0], dyad[1], type);
                        update_chain(1, change_stat);
                    }
                }
            }
        }
        // Second: Edges in overlapping subgraphs 
        type = 2;  
        for (int k = 0; k < network.block_count(); ++k) {
            for (int upd_num = 0; upd_num < burnin; ++upd_num) {
                u1 = runif(rand);
                u2 = runif(rand);
                select_dyad_overlap(k, dyad, u1, u2);
                compute_change_stat(dyad[0], dyad[1], change_stat, type);
                prop_prob = compute_prop_prob(dyad[0], dyad[1], change_stat, type);
                u = runif(rand);
                if (u <= prop_prob) {
                    if (network.is_edge(dyad[0], dyad[1], type)) {
                        network.delete_edge(dyad[0], dyad[1], type);
                        update_chain(-1, change_stat);
                    }
                    else {
                        network.add_edge(dyad[0], dyad[1], type);
                        update_chain(1, change_stat);
                    }
                }
            }
        }
        // Third: Between graph subgraphs
        for (int k = 0; k < (network.block_count() - 1); ++k) {
            for (int l = (k + 1); l < network.block_count(); ++l) {
                for (int upd_num = 0; upd_num < burnin; ++upd_num) {
                    u1 = runif(rand);
                    u2 = runif(rand);
                    select_dyad_between(k, l, dyad, u1, u2);
                    type = network.get_edge_type(dyad[0], dyad[1]);
                    compute_change_stat(dyad[0], dyad[1], change_stat, type);
                    prop_prob = compute_prop_prob(dyad[0], dyad[1], change_stat, type);
                    u = runif(rand);
                    if (u <= prop_prob) {
                        if (network.is_edge(dyad[0], dyad[1], type)) {
                            network.delete_edge(dyad[0], dyad[1], type);
                            update_chain(-1, change_stat);
                        }
                        else {
                            network.add_edge(dyad[0], dyad[1], type);
                            update_chain(1, change_stat);
                        }
                    }
                }
            }
        }
        // Next proceed to sample (after burn-ins)
        while (samp_num < num_samples) {
            // First: Unique subgraphs 
            type = 2;
            for (int k = 0; k < network.block_count(); ++k) {
                for (int upd_num = 0; upd_num < interval; ++upd_num) {
                    u1 = runif(rand);
                    u2 = runif(rand);
                    if (network.unique[k].size() <= 1) break;
                    select_dyad_unique(k, k, dyad, u1, u2);
                    compute_change_stat(dyad[0], dyad[1], change_stat, type);
                    prop_prob = compute_prop_prob(dyad[0], dyad[1], change_stat, type);
                    u = runif(rand);
                    if (u <= prop_prob) {
                        if (network.is_edge(dyad[0], dyad[1], type)) {
                            network.delete_edge(dyad[0], dyad[1], type);
                            update_chain(-1, change_stat);
                        }
                        else {
                            network.add_edge(dyad[0], dyad[1], type);
                            update_chain(1, change_stat);
                        }
                    }
                }
            }
            // Second: Edges in overlapping subgraphs 
            type = 2;
            for (int k = 0; k < network.block_count(); ++k) {
                for (int upd_num = 0; upd_num < interval; ++upd_num) {
                    u1 = runif(rand);
                    u2 = runif(rand);
                    select_dyad_overlap(k, dyad, u1, u2);
                    compute_change_stat(dyad[0], dyad[1], change_stat, type);
                    prop_prob = compute_prop_prob(dyad[0], dyad[1], change_stat, type);
                    u = runif(rand);
                    if (u <= prop_prob) {
                        if (network.is_edge(dyad[0], dyad[1], type)) {
                            network.delete_edge(dyad[0], dyad[1], type);
                            update_chain(-1, change_stat);
                        }
                        else {
                            network.add_edge(dyad[0], dyad[1], type);
                            update_chain(1, change_stat);
                        }
                    }
                }
            }
            // Third: Between graph subgraphs
            for (int k = 0; k < (network.block_count() - 1); ++k) {
                for (int l = (k + 1); l < network.block_count(); ++l) {
                    for (int upd_num = 0; upd_num < interval; ++upd_num) {
                        u1 = runif(rand);
                        u2 = runif(rand);
                        select_dyad_between(k, l, dyad, u1, u2);
                        type = network.get_edge_type(dyad[0], dyad[1]);
                        compute_change_stat(dyad[0], dyad[1], change_stat, type);
                        prop_prob = compute_prop_prob(dyad[0], dyad[1], change_stat, type);
                        u = runif(rand);
                        if (u <= prop_prob) {
                            if (network.is_edge(dyad[0], dyad[1], type)) {
                                network.delete_edge(dyad[0], dyad[1], type);
                                update_chain(-1, change_stat);
                            }
                            else {
                                network.add_edge(dyad[0], dyad[1], type);
                                update_chain(1, change_stat);
                            }
                        }
                    }
                }
            }
            // Record sample
            record_sample(samp_num);
            samp_num += 1;
        }
    }

};