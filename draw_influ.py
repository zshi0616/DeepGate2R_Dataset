import numpy as np 
import os 
import glob 
import random 
import time 
import networkx as nx
import matplotlib.pyplot as plt

from utils.utils import read_npz_file, split_into_ranges
import utils.circuit_utils as circuit_utils
from CondSim.wrapper import simulation

input_npz = '/Users/zhengyuanshi/studio/DeepGate2R/npz/graphs.npz'

LEVEL_RANGE = [0.2, 0.8]
PROB_THRESHOLD = 0.0    # 0.5 +- THRESHOLD
DEGREE_THRESHOLD = 0
MAX_SAMPLES = 5
COLOR_TYPE = 10

if __name__ == '__main__':
    graphs_02 = {}
    circuits = read_npz_file(input_npz)['circuits'].item()
    start_time = time.time()

    for cir_idx, cir_name in enumerate(circuits):
        x_data = circuits[cir_name]['x']
        edge_index = circuits[cir_name]['edge_index']
        if np.shape(edge_index)[0] == 2:
            edge_index = edge_index.T
        no_circuits = len(circuits)
        fanin_list, fanout_list = circuit_utils.get_fanin_fanout(x_data, edge_index)
        level_list = circuit_utils.get_level(x_data, fanin_list, fanout_list)
        pi_list, po_list = circuit_utils.get_pi_po(x_data, fanin_list, fanout_list)
        prob = simulation(x_data, edge_index, level_list, [[]])[0]
        
        # Sample gates
        cand_gate_list = []
        for level, arr in enumerate(level_list):
            if level < len(level_list) * LEVEL_RANGE[0] or level > len(level_list) * LEVEL_RANGE[1]:
                continue
            for idx in arr:
                if len(fanin_list[idx]) + len(fanout_list[idx]) < DEGREE_THRESHOLD:
                    continue
                cand_gate_list.append(idx)
        # print('Sampled {} conditions'.format(len(cond_list)))
        if len(cand_gate_list) == 0:
            continue
        
        # Sample condition 
        cond_list = []
        sample_gates = random.sample(cand_gate_list, min(MAX_SAMPLES, len(cand_gate_list)))
        for sample_k, gate in enumerate(sample_gates):
            if random.random() <= 0.5:
                cond_list.append([[gate, 0]])
            else:
                cond_list.append([[gate, 1]])
        
        # Simulation 
        prob_list = simulation(x_data, edge_index, level_list, cond_list)
        
        # Analysis 
        for prob_k, cond_prob in enumerate(prob_list):
            if len(cond_prob) == 0:
                continue
            err = np.abs(np.array(prob) - np.array(cond_prob))
            indices = split_into_ranges(err, err.min(), err.max(), COLOR_TYPE)
            cond_node = cond_list[prob_k][0][0]
            fanin_cone = circuit_utils.get_fanin_cone(fanin_list, [cond_node])
            fanout_cone = circuit_utils.get_fanout_cone(fanout_list, [cond_node])
            influt_area = []
            for fanin in fanin_cone:
                influt_area.append(fanin)
                if len(fanout_list[fanin]) > 0 and abs(x_data[fanin][2] - x_data[cond_node][2]) < 10:
                    sub_fo_cone = circuit_utils.get_fanout_cone(fanout_list, [fanin])
                    for fo_idx in sub_fo_cone:
                        if abs(x_data[fo_idx][2] - x_data[fanin][2]) < 10:
                            influt_area.append(fo_idx)
                influt_area = list(set(influt_area))
            
            # Network 
            G = nx.DiGraph()
            color_map = [''] * len(x_data)
            for idx, x_data_info in enumerate(x_data):
                G.add_node(idx)
                color_map[idx] = 'cornsilk'
            for src, x_data_info in enumerate(x_data):
                for dst in fanout_list[src]:
                    G.add_edge(src, dst)
                    
            # Color 
            colors = plt.get_cmap('Reds', COLOR_TYPE)
            for i in range(COLOR_TYPE):
                for idx in indices[i]:
                    color_map[idx] = colors(i)
            
            # Layout and Display 
            for idx in range(len(x_data)):
                G.nodes[idx]["level"] = x_data[idx][2]
            edgecolors = ['black'] * len(x_data)
            node_border_width = [0] * len(x_data)
            for idx in influt_area:
                node_border_width[idx] = 1.5
                edgecolors[idx] = 'blue'
            edgecolors[cond_node] = 'red'
            node_size = [150] * len(x_data)
            node_size[cond_node] = 500 
            for idx in influt_area:
                node_size[idx] = 200
            pos = nx.multipartite_layout(G, subset_key="level")

            # Draw
            plt.figure(figsize=(80, 40))
            nx.draw_networkx(G, pos=pos, font_size=5, 
                            node_color=color_map, node_size=node_size, 
                            edgecolors=edgecolors, linewidths=node_border_width, 
                            with_labels=True)
            pdf_filename = './fig/{}_{:}.pdf'.format(cir_name, prob_k)
            plt.savefig(pdf_filename)
            plt.close()
            
            print('Output: {}'.format(pdf_filename))
        