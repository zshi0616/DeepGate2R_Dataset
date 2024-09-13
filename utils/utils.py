import os 
import shlex
import subprocess
import time
import numpy as np
import torch

def read_npz_file(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data

def run_command(command):
    start_time = time.time()
    stdout = os.popen(command)
    stdout = stdout.read()
    end_time = time.time()

    run_time = end_time - start_time

    return stdout, run_time
    
def hash_arr(arr):
    p = 1543
    md = 6291469
    hash_res = 1
    tmp_arr = arr.copy()
    tmp_arr = np.sort(tmp_arr)
    for ele in tmp_arr:
        hash_res = (hash_res * p + ele) % md
    return hash_res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

def zero_normalization(x):
    mean_x = torch.mean(x)
    std_x = torch.std(x)
    z_x = (x - mean_x) / std_x
    return z_x

def split_into_ranges(lst, a, b, n):
    indices = [[] for _ in range(n)]
    step = (b - a) / n

    for i, num in enumerate(lst):
        if num <= a:
            indices[0].append(i)
        elif num >= b:
            indices[n-1].append(i)
        else:
            index = min(int((num - a) // step), n-1)
            indices[index].append(i)

    return indices