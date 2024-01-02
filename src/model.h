#pragma once
#include <algorithm>
#include <vector>
#include <string> 
#include <omp.h>
#include <iostream>
#include <stdlib.h>

#ifndef _helper_functions_
#define _helper_functions_
#include "helper_functions.h"
#endif

#ifndef _graph_functions_
#define _graph_functions_
#include "graph_functions.h"
#endif

#ifndef _change_stats_
#define _change_stats_
#include "change_stats.h"
#endif


using namespace std;


class mod {

public:
    vector<string> model_terms;
    vector<int> block_restriction;
    vector<double (*)(int i, int j, net_overlapping_blocks& network, int type, int block_restriction)> change_stat_funs;
    vector<double (*)(int i, int j, mlnet& network, int k, int l)> change_stat_funs_ml;
    vector<double (*)(net_overlapping_blocks& network, int block_restriction)> stat_funs;

public:
    mod(vector<string> mterms) {
        model_terms.resize(mterms.size());
        model_terms = mterms;
        block_restriction.resize(get_num_terms());
        change_stat_funs.resize(get_num_terms());
        change_stat_funs_ml.resize(get_num_terms());
        stat_funs.resize(get_num_terms());
        int iter = 0;
        int block_counter = 0;
        int node_counter = 0;
        for (string term : model_terms) {
            if (term == "edge") {
                change_stat_funs[iter] = cs_edge;
                stat_funs[iter] = gf_edge;
                block_restriction[iter] = -1;
                iter += 1;
            }
            if (term == "block_beta") {
                change_stat_funs[iter] = cs_block_beta;
                stat_funs[iter] = gf_block_beta;
                block_restriction[iter] = block_counter;
                block_counter += 1;
                iter += 1;
            }
            if (term == "brokerage") {
                change_stat_funs[iter] = cs_brokerage;
                stat_funs[iter] = gf_brokerage;
                block_restriction[iter] = -1;
                iter += 1;
            }
            if (term == "beta") {
                change_stat_funs[iter] = cs_beta;
                stat_funs[iter] = gf_beta;
                block_restriction[iter] = node_counter;
                iter += 1;
                node_counter += 1;
            }
            if (term == "ml_order2") { /// calculate change statistics for ml network with 2nd order cross-layer inetractions
                change_stat_funs_ml[iter] = cs_order2;
                iter += 1;
            }
            if (term == "ml_order3") { /// calculate change statistics for ml network with 3rd order cross-layer inetractions
                change_stat_funs_ml[iter] = cs_order3;
                iter += 1;
            }
            
        }
    }

    int get_num_terms() {
        int val = model_terms.size();
        return val;
    }

};
