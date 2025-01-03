import copy
import math
from dataclasses import asdict, make_dataclass
import mindspore
import mindspore.nn as nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell, MicroBatchInterleaved
from mindspore.dataset import GeneratorDataset, MindDataset
from mindspore.dataset.transforms import TypeCast
from mindformers.tools.register import MindFormerConfig
from mindformers.core.parallel_config import build_parallel_config
from mindformers import AutoConfig
from ppo_config import PPOConfig
from mindrlhf.utils.adam import AdamWeightDecayOp
from mindrlhf.utils.utils import LearningRate, FP32StateAdamWeightDecay
from mindrlhf.utils.dataset import IteratorStore
from mindrlhf.wrapper import TrainOneStepWithLossScaleCell, TrainPipelineWithLossScaleCell

__all__ = ['combine_config', 'init_configs']


def set_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
    def decay_filter(x): return 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    return group_params


def combine_config(ppo_config, model_config):
    config_temp = asdict(ppo_config)
    for k, v in model_config.items():
        if k not in config_temp:
            config_temp[k] = v
    config_temp['max_prompt_length'] = config_temp['seq_length'] - config_temp['max_decode_length']
    PPOConfig = make_dataclass("PPOConfig", [(key, type(value)) for key, value in config_temp.items()])
    return PPOConfig(**config_temp)


def init_configs(args=None):
    # 1. 初始化基础PPO配置
    ppo_config = PPOConfig()
    if args:
        ppo_config.mind_dataset_dir = args.dataset_dir
        ppo_config.sft_model_path = args.sft_model_path
        ppo_config.reward_model_path = args.reward_model_path
        ppo_config.critic_model_path = args.critic_model_path
        ppo_config.save_data_file = args.save_data_file
        ppo_config.align_type = args.align_type

    # 2. 创建基础模型参数
    base_params = {
        "batch_size": 2,
        "seq_length": 1024,
        "hidden_size": 5120,
        "num_layers": 38,
        "num_heads": 32,
        "vocab_size": 120000,
        "rms_norm_eps": 1.0e-5,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "ignore_token_id": -100,
        "hidden_dropout_prob": 0.1,
        "attention_dropout_prob": 0.1,
        "ffn_dim_multiplier": 12288,
        "compute_dtype": mindspore.float16,
        "layernorm_compute_type": mindspore.float32,
        "softmax_compute_type": mindspore.float16,
        "rotary_dtype": mindspore.float16,
        "param_init_type": mindspore.float16,
        "use_past": False,
        "pretrain_seqlen": 4096,
        "extend_method": "None",
        "use_flash_attention": True,
        "offset": 0,
        "use_past_shard": False,
        "checkpoint_name_or_path": "telechat_12b"
    }

    # 3. 为SFT模型添加生成相关配置
    sft_params = base_params.copy()
    sft_params.update({
        "repetition_penalty": 1,
        "max_decode_length": 512,
        "top_k": 3,
        "top_p": 1,
        "do_sample": False,
        "use_past": True  # SFT模型需要使用KV cache
    })

    # 4. 创建各个模型的配置
    sft_model_config = TelechatConfig(**sft_params)
    ref_model_config = TelechatConfig(**base_params)
    critic_model_config = TelechatConfig(**base_params)
    rm_model_config = TelechatConfig(**base_params)

    # 5. 设置batch size（如果使用use_past）
    if ppo_config.use_past:
        sft_model_config.batch_size = ppo_config.chunk_size
        ref_model_config.batch_size = ppo_config.chunk_size
        critic_model_config.batch_size = ppo_config.chunk_size
        rm_model_config.batch_size = ppo_config.chunk_size

    # 6. 更新PPO配置
    ppo_config.model_name = sft_model_config.model_name
    ppo_config = combine_config(ppo_config, sft_model_config)

    # 7. 打印配置信息
    print("[PPO Configure] is: ", ppo_config, flush=True)
    print("[ACT Configure] is: ", sft_model_config, sft_model_config.parallel_config, flush=True)
    print("[REF Configure] is: ", ref_model_config, ref_model_config.parallel_config, flush=True)
    print("[CRT Configure] is: ", critic_model_config, critic_model_config.parallel_config, flush=True)
    print("[RM Configure] is: ", rm_model_config, rm_model_config.parallel_config, flush=True)

    return ppo_config, sft_model_config, ref_model_config, critic_model_config, rm_model_config

def init_network_and_optimizer(trainer):
    '''init network and optimizer'''
    sft_model_config = trainer.sft_model_config
    ppo_config = trainer.ppo_config
    if sft_model_config.parallel_config.pipeline_stage > 1:
        print("pipeline cell")
        ppo_with_loss_net = PipelineCell(MicroBatchInterleaved(trainer.ppo_model,
                                                               ppo_config.micro_batch_interleaved),
                                         sft_model_config.parallel_config.micro_batch_num)
    else:
        print("non-pipeline cell")
        ppo_with_loss_net = trainer.ppo_model
    ppo_with_loss = _VirtualDatasetCell(ppo_with_loss_net)
    lr = LearningRate(learning_rate=ppo_config.start_lr, end_learning_rate=ppo_config.end_lr,
                      warmup_steps=ppo_config.warmup_step, decay_steps=ppo_config.decay_steps)
    params = ppo_with_loss.trainable_params()
    group_params = set_weight_decay(params)

    if ppo_config.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    elif ppo_config.opt_offload:
        optimizer = AdamWeightDecayOp(group_params, learning_rate=lr, eps=ppo_config.eps, beta1=ppo_config.beta1,
                                      beta2=ppo_config.beta2, param_init_type=sft_model_config.param_init_type)
    else:
        optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=lr, beta1=ppo_config.beta1,
                                             beta2=ppo_config.beta2, eps=ppo_config.eps)

    loss_scale_value = math.pow(2, 12)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value,
                                             scale_factor=2, scale_window=1000)

    if sft_model_config.parallel_config.pipeline_stage > 1:
        print("pipeline cell")
        ppo_with_grad = TrainPipelineWithLossScaleCell(ppo_with_loss, optimizer=optimizer, config=sft_model_config,
                                                       scale_update_cell=update_cell)
    else:
        print("non-pipeline cell")
        ppo_with_grad = TrainOneStepWithLossScaleCell(ppo_with_loss, optimizer=optimizer, config=sft_model_config,
                                                      scale_update_cell=update_cell, enable_global_norm=True)
    return ppo_with_grad


def init_ppo_dataset(trainer):
    ppo_config = trainer.ppo_config
    sft_model_config = trainer.sft_model_config
    column_names = ["query_tensors", "response_tensors", "logprobs",
                    "values", "rewards", "advantages", "returns",
                    "pretrain_ids", "loss_mask", "attention_mask"]
    if ppo_config.save_data_file and 'stages' in ppo_config.align_type:
        dataset = MindDataset(dataset_files=ppo_config.save_data_file, shuffle=False)
        dataset = dataset.project(columns=column_names)
    else:
        pipeline = IteratorStore(trainer.store)
        dataset = GeneratorDataset(pipeline, column_names=column_names)
    type_cast_op_int32 = TypeCast(mindspore.int32)
    type_cast_op_fp16 = TypeCast(mindspore.float16)
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="query_tensors")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="response_tensors")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="logprobs")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="values")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="rewards")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="advantages")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="returns")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="pretrain_ids")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="loss_mask")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="attention_mask")
    dataset = dataset.batch(batch_size=ppo_config.batch_size
                            * sft_model_config.parallel_config.data_parallel)
    return dataset

class TelechatConfig:
    """Telechat模型配置类"""
    def __init__(self, **kwargs):
        # 模型基础参数
        self.batch_size = kwargs.get('batch_size', 2)
        self.seq_length = kwargs.get('seq_length', 1024)
        self.hidden_size = kwargs.get('hidden_size', 5120)
        self.num_layers = kwargs.get('num_layers', 38)
        self.num_heads = kwargs.get('num_heads', 32)
        self.vocab_size = kwargs.get('vocab_size', 120000)
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1.0e-5)
        
        # Token IDs
        self.bos_token_id = kwargs.get('bos_token_id', 1)
        self.eos_token_id = kwargs.get('eos_token_id', 2)
        self.pad_token_id = kwargs.get('pad_token_id', 3)
        self.ignore_token_id = kwargs.get('ignore_token_id', -100)
        
        # Dropout配置
        self.hidden_dropout_prob = kwargs.get('hidden_dropout_prob', 0.1)
        self.attention_dropout_prob = kwargs.get('attention_dropout_prob', 0.1)
        
        # FFN配置
        self.ffn_dim_multiplier = kwargs.get('ffn_dim_multiplier', 12288)
        
        # 计算类型配置
        self.compute_dtype = kwargs.get('compute_dtype', mindspore.float16)
        self.layernorm_compute_type = kwargs.get('layernorm_compute_type', mindspore.float32)
        self.softmax_compute_type = kwargs.get('softmax_compute_type', mindspore.float16)
        self.rotary_dtype = kwargs.get('rotary_dtype', mindspore.float16)
        self.param_init_type = kwargs.get('param_init_type', mindspore.float16)
        
        # 模型功能配置
        self.use_past = kwargs.get('use_past', False)
        self.pretrain_seqlen = kwargs.get('pretrain_seqlen', 4096)
        self.extend_method = kwargs.get('extend_method', "None")
        self.use_flash_attention = kwargs.get('use_flash_attention', True)
        self.offset = kwargs.get('offset', 0)
        self.use_past_shard = kwargs.get('use_past_shard', False)
        
        # 模型标识
        self.checkpoint_name_or_path = kwargs.get('checkpoint_name_or_path', "telechat_12b")
        self.model_name = "telechat_12b"
        
        # 生成配置
        self.repetition_penalty = kwargs.get('repetition_penalty', 1)
        self.max_decode_length = kwargs.get('max_decode_length', 512)
        self.top_k = kwargs.get('top_k', 3)
        self.top_p = kwargs.get('top_p', 1)
        self.do_sample = kwargs.get('do_sample', False)
        
        # 并行配置
        self.parallel_config = self._init_parallel_config()
        
    def _init_parallel_config(self):
        """初始化并行配置"""
        class ParallelConfig:
            def __init__(self):
                self.data_parallel = 4
                self.model_parallel = 2
                self.pipeline_stage = 1
                self.use_seq_parallel = False
                self.micro_batch_num = 64
                self.vocab_emb_dp = True
                self.gradient_aggregation_group = 4
                
                # recompute配置
                class RecomputeConfig:
                    def __init__(self):
                        self.recompute = True
                        self.parallel_optimizer_comm_recompute = False
                        self.mp_comm_recompute = True
                        self.recompute_slice_activation = True
                self.recompute = RecomputeConfig()
        
        return ParallelConfig()

    def __str__(self):
        """返回配置的字符串表示"""
        return f"TelechatConfig(batch_size={self.batch_size}, seq_length={self.seq_length}, " \
               f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, " \
               f"num_heads={self.num_heads}, vocab_size={self.vocab_size}, ...)"