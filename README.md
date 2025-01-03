# telechat-12b rewardmodel实现，运行环境昇腾A100八卡，，，，
# 单卡运行可修改配置文件，多卡运行需先使用Ascend工具包中hccn_tool工具搭建网络环境，如果没有hccn_tool,请先完全安装Ascend工具包，或者配置hccn.conf
#创建 hccn.conf
sudo bash -c 'cat > /etc/hccn.conf << EOF
address_0=100.97.33.1
address_1=100.97.33.2
address_2=100.97.33.3
address_3=100.97.33.4
address_4=100.97.33.5
address_5=100.97.33.6
address_6=100.97.33.7
address_7=100.97.33.8
EOF'
# 启动训练运行train_reward.bash

pkill python
sleep 5
export PYTHONPATH=/work/code/sft/mindformers-r1.1.0/:$PYTHONPATH
export MS_DEV_SIDE_EFFECT_LOAD_ELIM=3  # 去除TensorMove
export MS_MEMORY_POOL_RECYCLE=1  # 内存优化
export GE_NOT_CUT=1   # 内存优化
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"  
#export MP_START_METHOD=spawn
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# 设置RANK_TABLE_FILE
RANK_TABLE_FILE=/work/code/sft/mindformers-r1.1.0/hccl_8p_01234567_10.223.136.24.json

# 启动训练
bash /work/code/sft/mindformers-r1.1.0/research/run_singlenode.sh \
"python /work/home/home/telechat_910B/train_reward.py \
--run_mode finetune" \
$RANK_TABLE_FILE [0,8] 8

其中export PYTHONPATH=/work/code/sft/mindformers-r1.1.0/:$PYTHONPATH为为环境添加环境变量
export MS_DEV_SIDE_EFFECT_LOAD_ELIM=3  # 去除TensorMove
export MS_MEMORY_POOL_RECYCLE=1  # 内存优化
export GE_NOT_CUT=1   # 内存优化
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"  
这些为使用flashtrain全参微调设计
/work/code/sft/mindformers-r1.1.0/research/run_singlenode.sh
run_singlenode.sh为一机多卡分配脚本

$RANK_TABLE_FILE [0,8] 8
表示调用1到8张卡

# 训练中遇到的问题：
#1.显卡之间通信，不同的容器要分别使用net.sh生成各自的“hccl_8p_01234567_10.223.136.24.json”\

#2.低版本MindSpore可能不支持Flash Attention，需要设置为False\
#3.TelechatForSequenceClassification使用Telechatmodel为父类，无法接收seq_length,通过日志信息调试后发现，没有参数传入freqs_mgr，最后将max_seq_length手动传入\
#4.算子预编译错误，多进程错误：\
#Exception in thread Thread-2:\
Traceback (most recent call last):
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/threading.py", line 980, in _bootstrap_inner\
    self.run()\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/threading.py", line 917, in run\
    self._target(*self._args, **self._kwargs)\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/multiprocessing/pool.py", line 513, in _handle_workers\
    cls._maintain_pool(ctx, Process, processes, pool, inqueue,\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/multiprocessing/pool.py", line 337, in _maintain_pool\
    Pool._repopulate_pool_static(ctx, Process, processes, pool,\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/multiprocessing/pool.py", line 326, in _repopulate_pool_static\
    w.start()\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/multiprocessing/process.py", line 121, in start\
    self._popen = self._Popen(self)\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/multiprocessing/context.py", line 291, in _Popen\
    return Popen(process_obj)\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/multiprocessing/popen_forkserver.py", line 35, in __init__\
    super().__init__(process_obj)\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/multiprocessing/popen_fork.py", line 19, in __init__\
    self._launch(process_obj)\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/multiprocessing/popen_forkserver.py", line 59, in _launch\
    self.pid = forkserver.read_signed(self.sentinel)\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/multiprocessing/forkserver.py", line 328, in read_signed\
    raise EOFError('unexpected EOF')\
EOFError: unexpected EOF\
Exception in thread Thread-1:\
Traceback (most recent call last):\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/threading.py", line 980, in _bootstrap_inner\
    self.run()\
  File "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/repository_manager/utils/multiprocess_util.py", line 91, in run\
    key, func, args, kwargs = self.task_q.get(timeout=TIMEOUT)\
  File "<string>", line 2, in get\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/multiprocessing/managers.py", line 810, in _callmethod\
    kind, result = conn.recv()\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/multiprocessing/connection.py", line 250, in recv\
    buf = self._recv_bytes()\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/multiprocessing/connection.py", line 414, in _recv_bytes\
    buf = self._recv(4)\
  File "/root/miniconda3/envs/mf1.1_ms2.3_py39/lib/python3.9/multiprocessing/connection.py", line 383, in _recv\
    raise EOFError\
EOFError\
