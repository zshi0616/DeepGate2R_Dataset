import numpy as np 
import os 
import glob 
import random 
import time 

from utils.utils import read_npz_file
import utils.circuit_utils as circuit_utils
from CondSim.wrapper import simulation

input_npz = '/Users/zhengyuanshi/studio/DeepGate2R/npz/graphs.npz'
output_path = './npz/cond_full_circuits.npz'

LEVEL_RANGE = [0.2, 0.8]
PROB_THRESHOLD = 0.4    # 0.5 +- THRESHOLD
DEGREE_THRESHOLD = 3
MAX_SAMPLES = 10

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
                if abs(prob[idx] - 0.5) < PROB_THRESHOLD:
                    continue
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
            cond_list.append([[gate, 0]])
            cond_list.append([[gate, 1]])
        
        # Simulation 
        prob_list = simulation(x_data, edge_index, level_list, cond_list)
        assert len(prob_list) == len(cond_list)
        type_mask = [0] * len(x_data)
        x_data = x_data.tolist()
        edge_index = edge_index.tolist()
        # Save as graph
        for prob_k, cond_prob in enumerate(prob_list):
            if len(cond_prob) == 0:
                continue
            # Structure analysis 
            cond_node = cond_list[prob_k][0][0]
            cond_phase = cond_list[prob_k][0][1]
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
                
            # Add AND nodes
            if cond_phase == 0:
                cond_gate = len(x_data)
                x_data.append([cond_gate, 2, x_data[cond_node][2] + 1])
                type_mask.append(1)
                edge_index.append([cond_node, cond_gate])
            else:
                cond_gate = cond_node
            for influt_idx in influt_area:
                if influt_idx == cond_gate:
                    continue
                new_add_gate = len(x_data)
                x_data.append([new_add_gate, 1, max(x_data[influt_idx][2], x_data[cond_gate][2]) + 1])
                type_mask.append(2)
                edge_index.append([influt_idx, new_add_gate])
                edge_index.append([cond_gate, new_add_gate])
            
        # Last simulation 
        fanin_list, fanout_list = circuit_utils.get_fanin_fanout(x_data, edge_index)
        level_list = circuit_utils.get_level(x_data, fanin_list, fanout_list)
        final_prob = simulation(x_data, edge_index, level_list, [[]])[0]
        graph = {}
        graph_name = '{}_{}'.format(cir_name, 'cond')
        graph = {
            'x': np.array(x_data, dtype=int),
            'edge_index': np.array(edge_index),
            'type_mask': np.array(type_mask),
            'prob': np.array(final_prob, dtype=float)
        }
        graphs_02[graph_name] = graph
        current_time = time.time() - start_time
        
        print('Save: {}, #Graphs: {:}, {:}/{:}={:.2f}%, ETA: {:.2f}s'.format(
            cir_name, len(graphs_02), cir_idx, no_circuits, cir_idx / no_circuits * 100, 
            current_time / (cir_idx + 1) * (no_circuits - cir_idx)
        ))
        
        if len(graphs_02) % 1000 == 0:
            tmp_output_path = output_path.replace('.npz', '_tmp.npz')
            np.savez(tmp_output_path, circuits=graphs_02)
            print(tmp_output_path)
            print(len(graphs_02))
            
        
    np.savez(output_path, circuits=graphs_02)
    print(output_path)
    print(len(graphs_02))
    print()
