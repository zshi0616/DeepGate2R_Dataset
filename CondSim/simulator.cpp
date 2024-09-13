#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <queue>
#include <time.h>
#define rep(p, q) for (int p=0; p<q; p++)
#define PI 0
#define AND 1
#define NOT 2
#define STATE_WIDTH 31

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 3) {
        cout << "Failed" << endl;
        return 1;
    }
    string in_filename = argv[1];
    string out_filename = argv[2];
    
    cout << "Read File: " << in_filename << endl;
    freopen(in_filename.c_str(), "r", stdin);
    int n, m;  // number of gates
    int no_patterns; 
    scanf("%d %d %d", &n, &m, &no_patterns);
    cout << "Number of gates: " << n << endl;

    // Graph
    vector<int> gate_list(n);
    vector<vector<int> > fanin_list(n);
    vector<vector<int> > fanout_list(n);
    vector<int> gate_levels(n);
    vector<int> pi_list;
    int max_level = 0;

    for (int k=0; k<n; k++) {
        int type, level;
        scanf("%d %d", &type, &level);
        gate_list[k] = type;
        gate_levels[k] = level;
        if (level > max_level) {
            max_level = level;
        }
        if (type == PI) {
            pi_list.push_back(k);
        }
    }
    vector<vector<int> > level_list(max_level+1);
    for (int k=0; k<n; k++) {
        level_list[gate_levels[k]].push_back(k);
    }
    for (int k=0; k<m; k++) {
        int fanin, fanout;
        scanf("%d %d", &fanin, &fanout);
        fanin_list[fanout].push_back(fanin);
        fanout_list[fanin].push_back(fanout);
    }

    int no_pi = pi_list.size();
    cout << "Number of PI: " << no_pi << endl;

    cout<<"Start Simulation"<<endl;
    // Simulation
    vector<vector<bool> > full_states(n); 
    int tot_clk = 0;
    int clk_cnt = 0; 
    int remain_patterns = no_patterns;

    while (remain_patterns > 0) {
        remain_patterns -= STATE_WIDTH;
        vector<uint64_t> states(n);
        // generate pi patterns 
        rep(k, no_pi) {
            int pi = pi_list[k];
            states[pi] = rand() % uint64_t(pow(2, STATE_WIDTH)); 
        }
        // Combination
        for (int l = 1; l < max_level+1; l++) {
            for (int gate: level_list[l]) {
                if (gate_list[gate] == AND) {
                    uint64_t res = (states[fanin_list[gate][0]] & states[fanin_list[gate][1]]); 
                    states[gate] = res;
                }
                else if (gate_list[gate] == NOT) {
                    uint64_t res = ~states[fanin_list[gate][0]]; 
                    states[gate] = res;
                }
            }
        }
        // Record
        rep(k, n) {
            rep(p, STATE_WIDTH) {
                bool tmp = (states[k] >> p) & 1; 
                full_states[k].push_back(tmp);
            }
        }
    }
    cout << "Initial Simulation Done" << endl;

    freopen(out_filename.c_str(), "w", stdout);

    // Conditions 
    int no_conditions, no_nodes;
    scanf("%d", &no_conditions);
    rep(cd_idx, no_conditions) {
        vector<int> cond_nodes; 
        vector<int> cond_types;
        vector<int> prob_list(n, 0);
        scanf("%d", &no_nodes);
        rep(k, no_nodes) {
            int node_idx, node_type;
            scanf("%d %d", &node_idx, &node_type);
            cond_nodes.push_back(node_idx);
            cond_types.push_back(node_type);
        }
        // Check
        int succ_pts = 0;
        bool is_succ = true;
        rep(p, no_patterns) {
            is_succ = true;
            rep(k, no_nodes) {
                if (full_states[cond_nodes[k]][p] != cond_types[k]) {
                    is_succ = false;
                    break;
                }
            }
            if (is_succ) {
                succ_pts += 1;
                rep(k, n) {
                    prob_list[k] += int(full_states[k][p]);
                }
            }
        }
        // Output
        printf("Cond: %d, patterns: %d\n", cd_idx, succ_pts);
        if (succ_pts != 0) {
            rep(k, n) {
                printf("%d %f\n", k, float(prob_list[k])/float(succ_pts));
            }
        }
    }
}