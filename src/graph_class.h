#pragma once
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <iostream>
#include <random>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <Rcpp.h>
#include "helper_functions.h"
using namespace std;


class mlnet {

public:
    int num_nodes;
    int num_layers;
    vector<vector<vector<int> > > adj; ///need to store edges in different layers

public:
    mlnet(int N, int K)
    {
        num_nodes = N;
        num_layers = K;
        adj.resize(num_nodes);
        for (int i = 0;i < num_nodes;++i) {
            adj[i].resize(num_nodes);
        }
    }

    int node_count() {
        return num_nodes;
    }

    int layer_count() {
        return num_layers;
    }

    void add_edge(int i, int j, int k) {
        adj[i][j].push_back(k);
        adj[j][i].push_back(k);
    }

    bool is_edge(int i, int j, int k) {
        bool val = false;
        val = is_in(adj[i][j], k);
        return val;
    }

    void delete_edge(int i, int j, int k) {
        vector<int>::iterator ind;
        ind = find(adj[i][j].begin(), adj[i][j].end(), k);
        swap(*ind, adj[i][j].back());
        adj[i][j].pop_back();
        ind = find(adj[j][i].begin(), adj[j][i].end(), k);
        swap(*ind, adj[j][i].back());
        adj[j][i].pop_back();
    }
};



class net_overlapping_blocks {

public:
    int num_nodes;
    int num_blocks;
    vector<vector<int> > adj;
    vector<vector<int> > adj_local;
    vector<vector<int> > adj_broker;
    vector<vector<int> > adj_global;
    vector<vector<int> > blocks;
    vector<vector<int> > unique;
    vector<vector<int> > overlap;
    vector<vector<vector<vector<int> > > > between;
    vector<int> primary_memberships;
    vector<vector<int> > node_neighborhoods;
    vector<vector<int> > node_blocks;
    vector<vector<bool> > block_overlap_graph;
    vector<vector<int> > edge_types;

public:
    net_overlapping_blocks(int N, vector<vector<int> > bmemb, int K) {
        int iter = 0;
        num_nodes = N;
        num_blocks = K;
        node_blocks = bmemb;
        block_overlap_graph.resize(num_blocks);
        for (int k = 0; k < num_blocks; ++k) {
            block_overlap_graph[k].resize(num_blocks);
            for (int l = 0; l < num_blocks; ++l) {
                if (l == k) {
                    block_overlap_graph[k][l] = true;
                }
                else {
                    block_overlap_graph[k][l] = false;
                }
            }
        }
        blocks.resize(num_blocks);
        unique.resize(num_blocks);
        overlap.resize(num_blocks);
        primary_memberships.resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            iter = 0;
            for (int k : bmemb[i]) {
                if (iter == 0) primary_memberships[i] = k;
                iter += 1;
                blocks[k].push_back(i);
                if (bmemb[i].size() == 1) {
                    unique[k].push_back(i);
                }
                else {
                    overlap[k].push_back(i);
                }
            }
            if (iter > 0) {
                for (int b1 : bmemb[i]) {
                    for (int b2 : bmemb[i]) {
                        block_overlap_graph[b1][b2] = true;
                        block_overlap_graph[b2][b1] = true;
                    }
                }
            }
        }
        adj.resize(num_nodes);
        adj_local.resize(num_nodes);
        adj_broker.resize(num_nodes);
        adj_global.resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            adj[i].reserve(500);
            adj_local[i].reserve(100);
            adj_broker[i].reserve(100);
            adj_global[i].reserve(400);
        }
        between.resize(num_blocks);
        for (int k = 0; k < num_blocks; ++k) {
            sort(blocks[k].begin(), blocks[k].end());
            between[k].resize(num_blocks);
            for (int l = 0; l < num_blocks; ++l) {
                if (k == l) continue;
                between[k][l].resize(2);
                set_difference(blocks[k].begin(), blocks[k].end(),
                    blocks[l].begin(), blocks[l].end(),
                    back_inserter(between[k][l][0]));
                set_difference(blocks[l].begin(), blocks[l].end(),
                    blocks[k].begin(), blocks[k].end(),
                    back_inserter(between[k][l][1]));
                sort(between[k][l][0].begin(), between[k][l][0].end());
                sort(between[k][l][1].begin(), between[k][l][1].end());
            }
        }
        node_neighborhoods.resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            for (int k : bmemb[i]) {
                for (int j : blocks[k]) {
                    if (i == j) continue;
                    node_neighborhoods[i].push_back(j);
                }
            }
            sort(node_neighborhoods[i].begin(), node_neighborhoods[i].end());
        }
        edge_types.resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            edge_types[i].resize(num_nodes);
            for (int j = 0; j < num_nodes; ++j) {
                if (i == j) {
                    edge_types[i][j] = -1;
                    continue;
                }
                if (are_same_block(i, j)) {
                    edge_types[i][j] = 2;
                }
                else if (do_neighborhoods_overlap(i, j)) {
                    edge_types[i][j] = 1;
                }
                else {
                    edge_types[i][j] = 0;
                }
            }
        }
    }

    
    void add_edge(int i, int j, int add_key) {
        adj[i].push_back(j);
        adj[j].push_back(i);
        if (add_key == 0) {
            adj_global[i].push_back(j);
            adj_global[j].push_back(i);
        }
        else if (add_key == 1) {
            adj_broker[i].push_back(j);
            adj_broker[j].push_back(i);
        }
        else if (add_key == 2) {
            adj_local[i].push_back(j);
            adj_local[j].push_back(i);
        }
    }

    
    void delete_edge(int i, int j, int del_key) {
        vector<int>::iterator ind;
        ind = find(adj[i].begin(), adj[i].end(), j);
        swap(*ind, adj[i].back());
        adj[i].pop_back();
        ind = find(adj[j].begin(), adj[j].end(), i);
        swap(*ind, adj[j].back());
        adj[j].pop_back();
        if (del_key == 0) {
            ind = find(adj_global[i].begin(), adj_global[i].end(), j);
            swap(*ind, adj_global[i].back());
            adj_global[i].pop_back();
            ind = find(adj_global[j].begin(), adj_global[j].end(), i);
            swap(*ind, adj_global[j].back());
            adj_global[j].pop_back();
        }
        else if (del_key == 1) {
            ind = find(adj_broker[i].begin(), adj_broker[i].end(), j);
            swap(*ind, adj_broker[i].back());
            adj_broker[i].pop_back();
            ind = find(adj_broker[j].begin(), adj_broker[j].end(), i);
            swap(*ind, adj_broker[j].back());
            adj_broker[j].pop_back();
        }
        else if (del_key == 2) {
            ind = find(adj_local[i].begin(), adj_local[i].end(), j);
            swap(*ind, adj_local[i].back());
            adj_local[i].pop_back();
            ind = find(adj_local[j].begin(), adj_local[j].end(), i);
            swap(*ind, adj_local[j].back());
            adj_local[j].pop_back();
        }
    }

   
    bool is_edge(int i, int j, int chk_key) {
        bool val = false;
        if (chk_key == -1) {
            val = is_in(adj[i], j);
        }
        else if (chk_key == 0) {
            val = is_in(adj_global[i], j);
        }
        else if (chk_key == 1) {
            val = is_in(adj_broker[i], j);
        }
        else if (chk_key == 2) {
            val = is_in(adj_local[i], j);
        }
        return val;
    }

    int block_count() {
        return num_blocks;
    }

    int node_count() {
        return num_nodes;
    }

    bool do_blocks_overlap(int block_id_1, int block_id_2) {
        return block_overlap_graph[block_id_1][block_id_2];
    }

    bool are_same_block(int i, int j) {
        bool result = false;
        for (int h : node_blocks[i]) {
            if (is_in(node_blocks[j], h)) {
                result = true;
                break;
            }
        }
        return result;
    }

    bool do_neighborhoods_overlap(int i, int j) {
        bool result = false;
        for (int h : node_neighborhoods[i]) {
            if (is_in(node_neighborhoods[j], h)) {
                result = true;
                break;
            }
        }
        return result;
    }

    int get_edge_type(int i, int j) {
        return edge_types[i][j];
    }

};