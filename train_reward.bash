# 设置环境变量

pkill python
sleep 5
export PYTHONPATH=/work/code/sft/mindformers-r1.1.0/:$PYTHONPATH
export MS_DEV_SIDE_EFFECT_LOAD_ELIM=3  # 去除TensorMove
export MS_MEMORY_POOL_RECYCLE=1  # 内存优化
export GE_NOT_CUT=1   # 内存优化
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"  
# export MP_START_METHOD=spawn
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# 设置RANK_TABLE_FILE
RANK_TABLE_FILE=/work/code/sft/mindformers-r1.1.0/hccl_8p_01234567_10.223.136.24.json

# 启动训练
bash /work/code/sft/mindformers-r1.1.0/research/run_singlenode.sh \
"python /work/home/home/telechat_910B/train_reward.py \
--run_mode finetune" \
$RANK_TABLE_FILE [0,8] 8