#pragma once
#include <algorithm>
#include <ctime>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <cstdarg>
#include "graph_functions.h" 
#include "graph_class.h"
using namespace std;

/*****************************
  Edge count change statistics
******************************/
double cs_edge(int i, int j, net_overlapping_blocks& network, int type, int block_restriction) {
    return 1;
}

/******************************************
  Within-Block Edge count change statistic
*******************************************/


/******************************************
  Between-Block Edge count change statistic
*******************************************/


/******************************************
  Beta change statistic
*******************************************/
double cs_beta(int i, int j, net_overlapping_blocks& network, int type, int block_restriction) {
    double count = 0;
    if (i == block_restriction) count += 1;
    if (j == block_restriction) count += 1;
    return count;
}


double cs_order2(int i, int j, mlnet& mlnetwork, int k, int l) {
    if (k != l) {
        if (mlnetwork.is_edge(i, j, l)) {
            return 1;
        }
        else return 0;
    }
    else return 1;
    
}

double cs_order3(int i, int j, mlnet& mlnetwork, int k, int l) {
    if (k != l) {
        if (mlnetwork.is_edge(i, j, l)) {
            return 1;
        }
        else return 0;
    }
    else return 1;

}



/******************************************
  Beta-Block Edge count change statistic
*******************************************/
double cs_block_beta(int i, int j, net_overlapping_blocks& network, int type, int block_restriction) {
    double count = 0;
    if (network.primary_memberships[i] == block_restriction) count += 1;
    if (network.primary_memberships[j] == block_restriction) count += 1;
    return count;
}

/******************************************
  Brokerage change statistic
*******************************************/
double cs_brokerage(int i, int j, net_overlapping_blocks& network, int type, int block_restriction) {

    if (type == 0) {
        return 0;
    }

    // double stat_change; 
    double count = 0;
    bool pass, check_ih, check_jh = false;
    bool ij_tran = false;
    bool broker_i = false;
    bool broker_j = false;

    if (type == 2) {
        if (network.node_blocks[i].size() > 1) broker_i = true; ///check if node i is a broker
        if (network.node_blocks[j].size() > 1) broker_j = true;
    }
    if (broker_i) {
        for (int h : network.adj_broker[j]) {
            pass = false;
            for (int check_i : network.adj_local[i]) {
                if (check_i == j) continue;
                if (check_i == h) {
                    pass = true;
                    break;
                }
            }
            if (!pass) continue;
            pass = false;
            for (int check_j : network.adj_local[j]) {
                if (check_j == i) continue;
                for (int check_h : network.adj_local[h]) {
                    if (check_h == i) continue;
                    if (check_j == check_h) {
                        pass = true;
                        break;
                    }
                }
                if (pass) break;
            }
            if (!pass) {
                count += 1;
                pass = false;
            }
        }
    }
    if (broker_j) {
        for (int h : network.adj_broker[i]) {
            pass = false;
            for (int check_j : network.adj_local[j]) {
                if (check_j == i) continue;
                if (check_j == h) {
                    pass = true;
                    break;
                }
            }
            if (!pass) continue;
            pass = false;
            for (int check_i : network.adj_local[i]) {
                if (check_i == j) continue;
                for (int check_h : network.adj_local[h]) {
                    if (check_h == j) continue;
                    if (check_i == check_h) {
                        pass = true;
                        break;
                    }
                }
                if (pass) break;
            }
            if (!pass) {
                count += 1;
                pass = false;
            }
        }
    }

    // Run over the local edges that i is connected to 
    for (int h : network.adj_local[i]) {
        pass = false;
        for (int check_j : network.adj_local[j]) {
            if (check_j == h) {
                pass = true;
                break;
            }
        }
        if (!pass) continue;
        if (!ij_tran) {
            ij_tran = true;
            count += 1;
        }
        pass = false;

        if (type == 2) {
            // Check if (i, h) is transitive independent of (i, j)
            check_ih = false;
            for (int s : network.adj_local[h]) {
                if ((s == i) || (s == j)) continue;
                for (int check_i : network.adj_local[i]) {
                    if (check_i == s) {
                        pass = true;
                        break;
                    }
                }
                if (!pass) continue;
                pass = false;
                check_ih = true;
                break;
            }
            if (!check_ih) count += 1;

            // Check if (j, h) is transitive independent of (i, j)
            check_jh = false;
            for (int s : network.adj_local[h]) {
                if ((s == i) || (s == j)) continue;
                for (int check_j : network.adj_local[j]) {
                    if (check_j == s) {
                        pass = true;
                        break;
                    }
                }
                if (!pass) continue;
                pass = false;
                check_jh = true;
                break;
            }
            if (!check_jh) count += 1;
        }
    }
    return(count);
}



/**********************************
  Transitive edge change statistic
***********************************/
