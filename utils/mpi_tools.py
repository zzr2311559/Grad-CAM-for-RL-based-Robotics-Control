import numpy as np
import os

# --- 关键修改：彻底移除 mpi4py 的导入 ---
# 我们手动定义单进程所需的函数，假装 MPI 存在但永远只有 1 个进程。

def mpi_fork(n, bind_to_core=False):
    """
    伪装的 mpi_fork。
    如果试图开启多进程，打印警告并忽略。
    """
    if n > 1:
        print("Warning: MPI fork disabled in this non-MPI version. Running in single process.")
    return

def msg(m, string=''):
    print(('Message from 0: %s \t '%string)+str(m))

def proc_id():
    """永远返回 0，表示主进程"""
    return 0

def allreduce(*args, **kwargs):
    return

def num_procs():
    """永远返回 1，表示只有一个进程"""
    return 1

def broadcast(x, root=0):
    pass

def mpi_op(x, op):
    return x

def mpi_sum(x):
    return x

def mpi_avg(x):
    return x
    
def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    单进程版的统计函数。
    直接计算 numpy 数组的 mean/std。
    """
    x = np.array(x, dtype=np.float32)
    
    # 防止空数组报错
    if len(x) == 0:
        if with_min_and_max:
            return 0.0, 0.0, 0.0, 0.0
        return 0.0, 0.0

    mean = np.mean(x)
    std = np.std(x)

    if with_min_and_max:
        return mean, std, np.min(x), np.max(x)
    return mean, std