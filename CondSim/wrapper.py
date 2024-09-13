import numpy as np 
import os 
import glob 
import time 
import threading
import random
import deepgate as dg
import shlex
import subprocess

def run_command(command, timeout=-1):
    try: 
        command_list = shlex.split(command)
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        start_time = time.time()
        while process.poll() is None:
            if timeout > 0 and time.time() - start_time > timeout:
                process.terminate()
                process.wait()
                raise TimeoutError(f"Command '{command}' timed out after {timeout} seconds")

            time.sleep(0.1)

        stdout, stderr = process.communicate()
        if len(stderr) > len(stdout):
            return str(stderr).split('\\n'), time.time() - start_time
        else:
            return str(stdout).split('\\n'), time.time() - start_time
    except TimeoutError as e:
        return e, -1
    

def simulation(
    x_data, edge_index, level_list, cond_list, 
    tmp_graph_path='', tmp_res_path='', 
    no_pts = 100000
):
    ''' 
        cond_list: list of the node conditions for simulation 
        [[[node1, cond_type], [node2, cond_type], ...], [[node1, cond_type]]]
        cond_type: 0/1
    '''
    
    if tmp_graph_path == '':
        tmp_graph_path = './tmp/tmp_graph_{}_{}_{}.txt'.format(
            time.strftime("%Y%m%d-%H%M%S"), threading.currentThread().ident, random.randint(0, 1000)
        )
    if tmp_res_path == '':
        tmp_res_path = './tmp/tmp_res_{}_{}_{}.txt'.format(
            time.strftime("%Y%m%d-%H%M%S"), threading.currentThread().ident, random.randint(0, 1000)
        )
    current_dir = os.path.dirname(__file__)
    
    # x_data: one hot or features 
    is_one_hot = True
    for x_data_info in x_data:
        if np.array(x_data_info).sum() != 1:
            is_one_hot = False
            break
    if is_one_hot:
        new_x_data = []
        for idx, x_data_info in enumerate(x_data):
            new_x_data.append([idx, np.argmax(x_data_info)])
        x_data = new_x_data
    
    # Parse forward_level 
    no_nodes = len(x_data)
    forward_level = [-1] * no_nodes
    for level, arr in enumerate(level_list):
        for node in arr:
            forward_level[node] = level
    
    # Write the graph to the file
    f = open(tmp_graph_path, 'w')
    f.write('{} {} {}\n'.format(len(x_data), len(edge_index), no_pts))
    for idx in range(no_nodes):
        f.write('{} {}\n'.format(int(x_data[idx][1]), forward_level[idx]))
    for edge in edge_index:
        f.write('{} {}\n'.format(edge[0], edge[1]))
    f.write('{}\n'.format(len(cond_list)))
    for cond in cond_list:
        f.write('{}\n'.format(len(cond)))
        for cond_arr in cond:
            f.write('{} {}\n'.format(cond_arr[0], cond_arr[1]))
    f.close()
    
    # Run the simulation
    simulator_path = os.path.join(current_dir, 'simulator')
    stdout, _ = run_command('{} {} {}'.format(simulator_path, tmp_graph_path, tmp_res_path))
    
    # Parse
    f = open(tmp_res_path, 'r')
    lines = f.readlines()
    f.close()
    tp = 0
    prob_list = []
    while tp < len(lines):
        line = lines[tp]
        if 'Cond' in line:
            cond_idx = line.replace('\n', '').replace(' ', '').split(',')[0].split(':')[-1]
            cond_idx = int(cond_idx)
            cond_pts = line.replace('\n', '').replace(' ', '').split(',')[-1].split(':')[-1]
            cond_pts = int(cond_pts)
            tp += 1 
            if cond_pts > 0:
                prob = [-1] * no_nodes
                for _ in range(no_nodes):
                    line = lines[tp]
                    tp += 1
                    arr = line.replace('\n', '').split(' ')
                    prob[int(arr[0])] = float(arr[1])
                prob_list.append(prob)
            else:
                prob_list.append([])
        else:
            tp += 1
    
    # Remove 
    os.remove(tmp_graph_path)
    os.remove(tmp_res_path)
    
    return prob_list
    
if __name__ == '__main__':
    x_data = [[0, 0], [1, 0], [2, 1]]
    edge_index = [[0, 2], [1, 2]]
    level_list = [[0, 1], [2]]
    cond_list = [[[0, 1], [1, 0]], [[2, 1]]]
    
    prob_list = simulation(x_data, edge_index, level_list, cond_list)
    print()