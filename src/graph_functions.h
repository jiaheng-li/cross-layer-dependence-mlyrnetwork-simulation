#pragma once
#include <algorithm>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include "graph_class.h"
using namespace std;


/*****************************
  Edge count statistic
******************************/
double gf_edge(net_overlapping_blocks& network, int block_restriction) {
    double edge_count = 0;
    for (int node_i = 0; node_i < network.node_count(); ++node_i) {
        edge_count += network.adj[node_i].size();
    }
    edge_count /= 2;
    return edge_count;
}


/**********************************
  Within-Block  edge statistic
***********************************/


/**********************************
  Between-Block  edge statistic
***********************************/


/**********************************
  Beta statistic
***********************************/
double gf_beta(net_overlapping_blocks& network, int block_restriction) {
    double count = 0;
    count = network.adj[block_restriction].size();
    return count;
}


/**********************************
  Beta Block  edge statistic
***********************************/
double gf_block_beta(net_overlapping_blocks& network, int block_restriction) {
    double count = 0;
    for (int node_i : network.blocks[block_restriction]) {
        if (network.primary_memberships[node_i] != block_restriction) continue;
        for (int node_j : network.adj[node_i]) {
            if (network.primary_memberships[node_j] == block_restriction) {
                count += 1;
            }
            else {
                count += 1;
            }
        }
    }
    return count;
}


/**********************************
  Brokerage statistic
***********************************/
double gf_brokerage(net_overlapping_blocks& network, int block_restriction) {
    double count = 0;
    bool pass = false;
    for (int i = 0; i < (network.node_count() - 1); ++i) {
        for (int j = (i + 1); j < network.node_count(); ++j) {
            if (network.get_edge_type(i, j) == 0) continue;
            if (!network.is_edge(i, j, network.get_edge_type(i, j))) continue;
            for (int h : network.adj_local[i]) {
                if (h == j) continue;
                for (int check_h : network.adj_local[j]) {
                    if (h == check_h) {
                        pass = true;
                        break;
                    }
                }
                if (pass) break;
            }
            if (pass) {
                count += 1;
                pass = false;
            }
        }
    }
    return count;
}

/**********************************
  Transitive edge statistic
***********************************/