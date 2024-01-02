#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

#include <random>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sstream>
#include <vector>


#ifndef _estimation_class_
#define _estimation_class_ 
#include "estimation_class.h"
#endif 


#ifndef _simulation_class_
#define _simulation_class_
#include "simulation_class.h"
#endif

using namespace std;
using namespace Rcpp;

// [[Rcpp::export]]
List rcpp_estimate_model_ml_Hway(IntegerMatrix RNETWORK,
    IntegerVector rsamp_num, IntegerVector rburnin, IntegerVector rinterval,
    IntegerVector rmodel_dim, StringVector model_terms,
    IntegerVector rnum_nodes, IntegerVector rnum_layers, IntegerVector rhighest_order, IntegerVector random_seeds, NumericVector g) {

    int samp_num = rsamp_num[0];
    int burnin = rburnin[0];
    int interval = rinterval[0];
    int model_dim = rmodel_dim[0];
    int num_nodes = rnum_nodes[0];
    int num_layers = rnum_layers[0];
    int highest_order = rhighest_order[0];
    int random_seed = random_seeds[0];
    double gy = g[0];

    vector<string> mterms;
    mterms.resize(model_dim);
    for (int p = 0; p < model_dim; ++p) {
        mterms.at(p) = model_terms[p];
    }

    est_ml_Hway est_obj(samp_num, burnin, interval, model_dim, mterms, num_nodes, num_layers, highest_order, random_seed, gy);

    int node_i, node_j, layer_k;
    for (int i = 0; i < RNETWORK.nrow(); ++i) {
        node_i = RNETWORK(i, 0) - 1;
        node_j = RNETWORK(i, 1) - 1;
        layer_k = RNETWORK(i, 2) - 1;
        est_obj.obs_net_ml.add_edge(node_i, node_j, layer_k);
    }
    est_obj.compute_initial_estimate();
    NumericVector Rtheta_est(est_obj.theta_est.size());
    for (int p = 0; p < Rtheta_est.length(); ++p) {
        Rtheta_est(p) = est_obj.theta_est.at(p);
    }
    List return_list = List::create(Named("theta_est") = Rtheta_est,
        Named("model_terms") = model_terms);

    Rcpp::Rcout << "\n" << flush;
    return return_list;

}




// [[Rcpp::export]]
List rcpp_estimate_model_ml(IntegerMatrix RNETWORK, NumericVector rNR_tol, IntegerVector rNR_max, IntegerVector rMCMLE_max,
    IntegerVector rsamp_num, IntegerVector rburnin, IntegerVector rinterval,
    IntegerVector rmodel_dim, StringVector model_terms,
    IntegerVector rnum_nodes, IntegerVector rnum_layers, LogicalVector rcheck_chull, IntegerVector random_seeds, NumericVector g) {

    int NR_max = rNR_max[0];
    int MCMLE_max = rMCMLE_max[0];
    double NR_tol = rNR_tol[0];
    int samp_num = rsamp_num[0];
    int burnin = rburnin[0];
    int interval = rinterval[0];
    int model_dim = rmodel_dim[0];
    int num_nodes = rnum_nodes[0];
    int num_layers = rnum_layers[0];
    bool check_chull = rcheck_chull[0];
    int random_seed = random_seeds[0];
    double gy = g[0];
    vector<vector<int> > MEMB;
    MEMB.resize(num_nodes);
   
    vector<string> mterms;
    mterms.resize(model_dim);
    for (int p = 0; p < model_dim; ++p) {
        mterms.at(p) = model_terms[p];
    }

    est_ml est_obj(samp_num, burnin, interval, model_dim, mterms, num_nodes, num_layers, random_seed, NR_max, NR_tol, MCMLE_max, check_chull, gy);
    // Copy observed network to estimation object
    int node_i, node_j, layer_k;
    for (int i = 0; i < RNETWORK.nrow(); ++i) {
        node_i = RNETWORK(i, 0) - 1;
        node_j = RNETWORK(i, 1) - 1;
        layer_k = RNETWORK(i, 2) - 1;
        est_obj.obs_net_ml.add_edge(node_i, node_j, layer_k);
    }

   

    est_obj.compute_initial_estimate();

    // Create return list
    NumericVector Rtheta_est(est_obj.theta_est.size());
    for (int p = 0; p < Rtheta_est.length(); ++p) {
        Rtheta_est(p) = est_obj.theta_est.at(p);
    }
    List return_list = List::create(Named("theta_est") = Rtheta_est,
        Named("model_terms") = model_terms);

    Rcpp::Rcout << "\n" << flush;
    return return_list;
}



// [[Rcpp::export]]
List rcpp_estimate_model(IntegerMatrix RNETWORK, NumericVector rNR_tol, IntegerVector rNR_max, IntegerVector rMCMLE_max,
    IntegerVector rsamp_num, IntegerVector rburnin, IntegerVector rinterval,
    IntegerVector rmodel_dim, StringVector model_terms,
    IntegerVector rnum_nodes, List bmemb, IntegerVector rnum_blocks, LogicalVector rcheck_chull) {

    int NR_max = rNR_max[0];
    int MCMLE_max = rMCMLE_max[0];
    double NR_tol = rNR_tol[0];
    int samp_num = rsamp_num[0];
    int burnin = rburnin[0];
    int interval = rinterval[0];
    int model_dim = rmodel_dim[0];
    int num_nodes = rnum_nodes[0];
    int num_blocks = rnum_blocks[0];
    bool check_chull = rcheck_chull[0];
    vector<vector<int> > MEMB;
    MEMB.resize(num_nodes);

    for (int i = 0; i < bmemb.length(); ++i) {
        IntegerVector cur_memb = bmemb[i];
        MEMB.at(i).resize(cur_memb.length());
        for (int j = 0; j < cur_memb.length(); ++j) {
            MEMB.at(i).at(j) = cur_memb[j];
        }
    }
    vector<string> mterms;
    mterms.resize(model_dim);
    for (int p = 0; p < model_dim; ++p) {
        mterms.at(p) = model_terms[p];
    }
    est est_obj(samp_num, burnin, interval, model_dim, mterms, num_nodes, MEMB, num_blocks, NR_max, NR_tol, MCMLE_max, check_chull);
    // Copy observed network to estimation object
    int node_i, node_j;
    for (int i = 0; i < RNETWORK.nrow(); ++i) {
        node_i = RNETWORK(i, 0) - 1;
        node_j = RNETWORK(i, 1) - 1;
        est_obj.obs_net.add_edge(node_i, node_j, est_obj.obs_net.get_edge_type(node_i, node_j));
    }

    
    Rcpp::Rcout << "\nNetwork:\n";
    for (int i = 0; i < num_nodes; ++i) {
      Rcpp::Rcout << "\n" << i << " -> ";
      for (int j : est_obj.obs_net.adj[i]) {
        Rcpp::Rcout << j << " ";
      }
    }

    Rcpp::Rcout << "\nNetwork-local:\n";
    for (int i = 0; i < num_nodes; ++i) {
      Rcpp::Rcout << "\n" << i << " -> ";
      for (int j : est_obj.obs_net.adj_local[i]) {
        Rcpp::Rcout << j << " ";
      }
    }

    Rcpp::Rcout << "\nNetwork-broker:\n";
    for (int i = 0; i < num_nodes; ++i) {
      Rcpp::Rcout << "\n" << i << " -> ";
      for (int j : est_obj.obs_net.adj_broker[i]) {
        Rcpp::Rcout << j << " ";
      }
    }
    Rcpp::Rcout << "\nNetwork-global:\n";
    for (int i = 0; i < num_nodes; ++i) {
      Rcpp::Rcout << "\n" << i << " -> ";
      for (int j : est_obj.obs_net.adj_global[i]) {
        Rcpp::Rcout << j << " ";
      }
    }

    est_obj.compute_initial_estimate();
    est_obj.compute_mcmle();
    // Create return list
    NumericVector Rtheta_est(est_obj.theta_est.size());
    for (int p = 0; p < Rtheta_est.length(); ++p) {
        Rtheta_est(p) = est_obj.theta_est.at(p);
    }
    List return_list = List::create(Named("theta_est") = Rtheta_est,
        Named("model_terms") = model_terms);

    Rcpp::Rcout << "\n" << flush;
    //cout << "\nmade it to estimation check point 6\n";
    return return_list;
}



//' @title rcpp_simulate
//' @description
//' Simulate a network model
//' @name rcpp_simulate
//' @param
//' @examples
//' rcp
// p_simulate(1,1,1,4,c("edge","brokerage","beta","block_beta"),5,list(c(0,2),c(0,1),c(1),c(2),c(2)),3,c(0.5,0.5,0.5,0.5))
//' 
//' @export
// [[Rcpp::export]]
List rcpp_simulate(IntegerVector rsamp_num, IntegerVector rburnin, IntegerVector rinterval,
    IntegerVector rmodel_dim, StringVector model_terms,
    IntegerVector rnum_nodes, List bmemb, IntegerVector rnum_blocks,
    NumericVector rtheta) {


    int samp_num = rsamp_num[0];
    int burnin = rburnin[0];
    int interval = rinterval[0];
    int model_dim = rmodel_dim[0];
    int num_nodes = rnum_nodes[0];
    int num_blocks = rnum_blocks[0];

    vector<vector<int> > MEMB;
    MEMB.resize(num_nodes);
    for (int i = 0; i < bmemb.length(); ++i) {
        IntegerVector cur_memb = bmemb[i];
        MEMB.at(i).resize(cur_memb.length());
        for (int j = 0; j < cur_memb.length(); ++j) {
            MEMB.at(i).at(j) = cur_memb[j];
        }
    }
    vector<double> theta;
    theta.resize(model_dim);
    vector<string> mterms;
    mterms.resize(model_dim);
    for (int p = 0; p < model_dim; ++p) {
        mterms.at(p) = model_terms[p];
        theta.at(p) = rtheta[p];
    }

    sim sim_obj(samp_num, burnin, interval, model_dim, mterms, num_nodes, MEMB, num_blocks);
    sim_obj.set_theta(theta);
    sim_obj.simulate();
    sim_obj.compute_sample_mean();
    Rcpp::Rcout << "\n\nEstimated mean = (";
    for (int p = 0; p < model_dim; ++p) {
        if (p == (model_dim - 1)) {
            Rcpp::Rcout << sim_obj.mean[p] << ")." << flush;
        }
        else {
            Rcpp::Rcout << sim_obj.mean[p] << ", ";
        }
    }

    Rcpp::Rcout << "\n\nChain state = (";
    for (int p = 0; p < model_dim; ++p) {
        if (p == (model_dim - 1)) {
            Rcpp::Rcout << sim_obj.chain_state[p] << ")." << flush;
        }
        else {
            Rcpp::Rcout << sim_obj.chain_state[p] << ", ";
        }
    }

    // Make edge list
    int edge_count = 0;
    int node_j;
    for (int i = 0; i < sim_obj.network.node_count(); ++i) {
        edge_count += sim_obj.network.adj[i].size();
    }
    edge_count /= 2;
    NumericMatrix elist(edge_count, 2);
    int iter = 0;
    for (int node_i = 0; node_i < sim_obj.network.node_count(); ++node_i) {
        for (int loc = 0; loc < sim_obj.network.adj[node_i].size(); ++loc) {
            node_j = sim_obj.network.adj[node_i][loc];
            if (node_i > node_j) continue;
            elist(iter, 0) = node_i + 1;
            elist(iter, 1) = node_j + 1;
            iter += 1;
        }
    }

    // Create return list
    List return_list = List::create(Named("elist") = elist);

    return return_list;
}

//' @title rcpp_simulate_ml
//' @description
//' Simulate a multilayer network model
//' @name rcpp_simulate_ml
//' @param
//' @examples
//' rcpp_simulate_ml(1,100,1,2,c("withinlayer","crosslayer"),3,2,c(0.4,0.7))
//' 
//' @export
// [[Rcpp::export]]
List rcpp_simulate_ml(IntegerVector rsamp_num, IntegerVector rburnin, IntegerVector rinterval,
    IntegerVector rmodel_dim, StringVector model_terms,
    IntegerVector rnum_nodes, IntegerVector rnum_layers,
    NumericVector rtheta, int rand_seed, NumericVector g) {

    
    int samp_num = rsamp_num[0];
    int burnin = rburnin[0];
    int interval = rinterval[0];
    int model_dim = rmodel_dim[0];
    int num_nodes = rnum_nodes[0];
    int num_layers = rnum_layers[0];
    int random_seed = rand_seed;
    double gy = g[0];


    vector<double> theta;
    theta.resize(model_dim);
    vector<string> mterms;
    mterms.resize(model_dim);
    for (int p = 0; p < model_dim; ++p) {
        mterms.at(p) = model_terms[p];
        theta.at(p) = rtheta[p];
    }

    sim_ml sim_obj(samp_num, burnin, interval, model_dim, mterms, num_nodes, num_layers, random_seed, gy);
    sim_obj.set_theta(theta);
    sim_obj.simulate_ml();
    sim_obj.compute_sample_mean();
   
    
    // print network output for estimation
    int edge_count = 0;
    for (int i = 0; i < sim_obj.mlnetwork.node_count(); ++i) {
        for (int j = i+1; j < sim_obj.mlnetwork.node_count(); ++j) {
            edge_count += sim_obj.mlnetwork.adj[i][j].size();
        }
    }

    
    NumericMatrix elist(edge_count, 3);
    int iter = 0;
    for (int node_i = 0; node_i < sim_obj.mlnetwork.node_count(); ++node_i) {
        for (int node_j = node_i+1; node_j < sim_obj.mlnetwork.node_count(); ++node_j) {
            for (int loc = 0; loc < sim_obj.mlnetwork.adj[node_i][node_j].size(); ++loc) {
              
                elist(iter, 0) = node_i + 1;
                elist(iter, 1) = node_j + 1;
                elist(iter, 2) = sim_obj.mlnetwork.adj[node_i][node_j][loc] + 1;
                iter += 1;
            }
        }
        
    }
    
    
    // Create return list
    List return_list = List::create(Named("elist") = elist);

    return return_list;
}


// [[Rcpp::export]]
List rcpp_simulate_ml_Hway(IntegerVector rsamp_num, IntegerVector rburnin, IntegerVector rinterval,
    IntegerVector rmodel_dim, StringVector model_terms,
    IntegerVector rnum_nodes, IntegerVector rnum_layers, IntegerVector highest_order,
    NumericVector rtheta, int rand_seed, NumericVector g) {


    int samp_num = rsamp_num[0];
    int burnin = rburnin[0];
    int interval = rinterval[0];
    int model_dim = rmodel_dim[0];
    int num_nodes = rnum_nodes[0];
    int num_layers = rnum_layers[0];
    int random_seed = rand_seed;
    double gy = g[0];
    int H = highest_order[0];

    vector<double> theta;
    theta.resize(model_dim);
    vector<string> mterms;
    mterms.resize(model_dim);
    for (int p = 0; p < model_dim; ++p) {
        mterms.at(p) = model_terms[p];
        theta.at(p) = rtheta[p];
    }

    sim_ml_Hway sim_obj(samp_num, burnin, interval, model_dim, mterms, num_nodes, num_layers, H, random_seed, gy);
    sim_obj.set_theta(theta);
    sim_obj.simulate_ml();
    sim_obj.compute_sample_mean();
  
    // print network output for estimation
    int edge_count = 0;
    for (int i = 0; i < sim_obj.mlnetwork.node_count(); ++i) {
        for (int j = i + 1; j < sim_obj.mlnetwork.node_count(); ++j) {
            edge_count += sim_obj.mlnetwork.adj[i][j].size();
        }
    }


    NumericMatrix elist(edge_count, 3);
    int iter = 0;
    for (int node_i = 0; node_i < sim_obj.mlnetwork.node_count(); ++node_i) {
        for (int node_j = node_i + 1; node_j < sim_obj.mlnetwork.node_count(); ++node_j) {
            for (int loc = 0; loc < sim_obj.mlnetwork.adj[node_i][node_j].size(); ++loc) {

                elist(iter, 0) = node_i + 1;
                elist(iter, 1) = node_j + 1;
                elist(iter, 2) = sim_obj.mlnetwork.adj[node_i][node_j][loc] + 1;
                iter += 1;
            }
        }

    }

    List return_list = List::create(Named("elist") = elist);

    return return_list;
}


//' @title rcpp_simulate_ml_Hway_suffstats
//' @description
//' Simulate a multilayer network model and print out sufficient statistics
//' @name rcpp_simulate_ml_Hway_suffstats
//' @param
//' @examples
//' rcpp_simulate_ml_Hway_suffstats(1,100,1,2,c("withinlayer","crosslayer"),3,2,2, c(0.4,0.7),123,1)
//' 
//' @export
// [[Rcpp::export]]
List rcpp_simulate_ml_Hway_suffstats(IntegerVector rsamp_num, IntegerVector rburnin, IntegerVector rinterval,
    IntegerVector rmodel_dim, StringVector model_terms,
    IntegerVector rnum_nodes, IntegerVector rnum_layers, IntegerVector highest_order,
    NumericVector rtheta, int rand_seed, NumericVector g) {


    int samp_num = rsamp_num[0];
    int burnin = rburnin[0];
    int interval = rinterval[0];
    int model_dim = rmodel_dim[0];
    int num_nodes = rnum_nodes[0];
    int num_layers = rnum_layers[0];
    int random_seed = rand_seed;
    double gy = g[0];
    int H = highest_order[0];

    vector<double> theta;
    theta.resize(model_dim);
    vector<string> mterms;
    mterms.resize(model_dim);
    for (int p = 0; p < model_dim; ++p) {
        mterms.at(p) = model_terms[p];
        theta.at(p) = rtheta[p];
    }

    sim_ml_Hway sim_obj(samp_num, burnin, interval, model_dim, mterms, num_nodes, num_layers, H, random_seed, gy);
    sim_obj.set_theta(theta);
    sim_obj.simulate_ml();
    sim_obj.compute_sample_mean();
   

    
    // Print some statistics
    int m;
    int i = 0;
    m = sim_obj.netCount_samp.size() + sim_obj.nCr(sim_obj.mlnetwork.layer_count(), 2) + 1;  // +1 to add 3-way count.
    NumericMatrix elist(m, 3);
    double count_avg;
    for (int iter = 0;iter < sim_obj.mlnetwork.layer_count();++iter) {
        for (int iter2 = iter; iter2 < sim_obj.mlnetwork.layer_count(); ++iter2) {

            elist(i, 0) = iter;
            elist(i, 1) = iter2;
            if (iter == iter2) {
                count_avg = (double)accumulate(sim_obj.netCount_samp[iter].begin(), sim_obj.netCount_samp[iter].end(), 0) / (double)sim_obj.num_samples;

            }
            else {
                count_avg = (double)accumulate(sim_obj.crossLayerCount_samp[iter][iter2].begin(), sim_obj.crossLayerCount_samp[iter][iter2].end(), 0) / (double)sim_obj.num_samples;
            }
            elist(i, 2) = count_avg;
            ++i;

        }

    }

    // For 3-way count. Note that even 3-way parameter is not specified we can always count this statistic.
    count_avg = (double)accumulate(sim_obj.threewayCount_samp.begin(), sim_obj.threewayCount_samp.end(), 0) / (double)sim_obj.num_samples;
    elist(i, 0) = 3;
    elist(i, 1) = 3;
    elist(i, 2) = count_avg;
    


    // Create return list
    List return_list = List::create(Named("elist") = elist);

    return return_list;
}


//' @title rcpp_simulate_ml_suffstats
//' @description
//' Simulate a multilayer network model and print out sufficient statistics
//' @name rcpp_simulate_ml_suffstats
//' @param
//' @examples
//' rcpp_simulate_ml_suffstats(1,100,1,2,c("withinlayer","crosslayer"),3,2, c(0.4,0.7),123,1)
//' 
//' @export
// [[Rcpp::export]]
List rcpp_simulate_ml_suffstats(IntegerVector rsamp_num, IntegerVector rburnin, IntegerVector rinterval,
    IntegerVector rmodel_dim, StringVector model_terms,
    IntegerVector rnum_nodes, IntegerVector rnum_layers, 
    NumericVector rtheta, int rand_seed, NumericVector g) {


    int samp_num = rsamp_num[0];
    int burnin = rburnin[0];
    int interval = rinterval[0];
    int model_dim = rmodel_dim[0];
    int num_nodes = rnum_nodes[0];
    int num_layers = rnum_layers[0];
    int random_seed = rand_seed;
    double gy = g[0];
    

    vector<double> theta;
    theta.resize(model_dim);
    vector<string> mterms;
    mterms.resize(model_dim);
    for (int p = 0; p < model_dim; ++p) {
        mterms.at(p) = model_terms[p];
        theta.at(p) = rtheta[p];
    }

    sim_ml sim_obj(samp_num, burnin, interval, model_dim, mterms, num_nodes, num_layers, random_seed, gy);
    sim_obj.set_theta(theta);
    sim_obj.simulate_ml();
    sim_obj.compute_sample_mean();
   



    // Print some statistics
    int m;
    int i = 0;
    m = sim_obj.netCount_samp.size() + sim_obj.nCr(sim_obj.mlnetwork.layer_count(), 2) + 1;  // +1 to add 3-way count.
    NumericMatrix elist(m, 3);
    double count_avg;
    for (int iter = 0; iter < sim_obj.mlnetwork.layer_count(); ++iter) {
        for (int iter2 = iter; iter2 < sim_obj.mlnetwork.layer_count(); ++iter2) {

            elist(i, 0) = iter;
            elist(i, 1) = iter2;
            if (iter == iter2) {
                count_avg = (double)accumulate(sim_obj.netCount_samp[iter].begin(), sim_obj.netCount_samp[iter].end(), 0) / (double)sim_obj.num_samples;

            }
            else {
                count_avg = (double)accumulate(sim_obj.crossLayerCount_samp[iter][iter2].begin(), sim_obj.crossLayerCount_samp[iter][iter2].end(), 0) / (double)sim_obj.num_samples;
            }
            elist(i, 2) = count_avg;
            ++i;

        }

    }

    // For 3-way count. Note that even 3-way parameter is not specified we can always count this statistic.
    count_avg = (double)accumulate(sim_obj.threewayCount_samp.begin(), sim_obj.threewayCount_samp.end(), 0) / (double)sim_obj.num_samples;
    elist(i, 0) = 3;
    elist(i, 1) = 3;
    elist(i, 2) = count_avg;



    // Create return list
    List return_list = List::create(Named("elist") = elist);

    return return_list;
}



//' @title rcpp_exact_simulate_ml
//' @description
//' Simulate a multilayer network model by its joint density
//' @name rcpp_exact_simulate_ml
//' @param
//' @examples
//' rcpp_exact_simulate_ml(1,1,3,7,rep("ml_order2",7),10,3,c(0,0.693,0.693,0,0.693,0,0),23333)
//'
//' @export
// [[Rcpp::export]]
List rcpp_exact_simulate_ml(IntegerVector rsamp_num, IntegerVector rburnin, IntegerVector rinterval,
    IntegerVector rmodel_dim, StringVector model_terms,
    IntegerVector rnum_nodes, IntegerVector rnum_layers,
    NumericVector rtheta, int rand_seed) {


    int samp_num = rsamp_num[0];
    int burnin = rburnin[0];
    int interval = rinterval[0];
    int model_dim = rmodel_dim[0];
    int num_nodes = rnum_nodes[0];
    int num_layers = rnum_layers[0];
    int random_seed = rand_seed;


    vector<double> theta;
    theta.resize(model_dim);
    vector<string> mterms;
    mterms.resize(model_dim);
    for (int p = 0; p < model_dim; ++p) {
        mterms.at(p) = model_terms[p];
        theta.at(p) = rtheta[p];
    }

    sim_ml_exact sim_obj_exact(samp_num, burnin, interval, model_dim, mterms, num_nodes, num_layers, random_seed);
    sim_obj_exact.set_theta(theta);
    sim_obj_exact.simulate_ml_exact();
   
    int edge_count = 0;
    for (int i = 0; i < sim_obj_exact.mlnetwork.node_count(); ++i) {
        for (int j = i + 1; j < sim_obj_exact.mlnetwork.node_count(); ++j) {
            edge_count += sim_obj_exact.mlnetwork.adj[i][j].size();
        }
    }

   

    
    // Print some statistics
    int m;
    int i = 0;
    m = sim_obj_exact.netCount_samp_exact.size() + sim_obj_exact.nCr(sim_obj_exact.mlnetwork.layer_count(), 2) + 1;
    NumericMatrix elist(m, 3);
    double count_avg;
    for (int iter = 0;iter < sim_obj_exact.mlnetwork.layer_count();++iter) {
        for (int iter2 = iter; iter2 < sim_obj_exact.mlnetwork.layer_count(); ++iter2) {

            elist(i, 0) = iter;
            elist(i, 1) = iter2;
            if (iter == iter2) {
                count_avg = (double)accumulate(sim_obj_exact.netCount_samp_exact[iter].begin(), sim_obj_exact.netCount_samp_exact[iter].end(), 0) / (double)sim_obj_exact.num_samples;

            }
            else {
                count_avg = (double)accumulate(sim_obj_exact.crossLayerCount_samp_exact[iter][iter2].begin(), sim_obj_exact.crossLayerCount_samp_exact[iter][iter2].end(), 0) / (double)sim_obj_exact.num_samples;
            }
            elist(i, 2) = count_avg;
            ++i;

        }

    }
    count_avg = (double)accumulate(sim_obj_exact.threewayCount_samp.begin(), sim_obj_exact.threewayCount_samp.end(), 0) / (double)sim_obj_exact.num_samples;
    elist(6, 0) = 3;
    elist(6, 1) = 3;
    elist(6, 2) = count_avg;


    
    // Create return list
    List return_list = List::create(Named("elist") = elist);

    return return_list;
}
