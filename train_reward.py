
import os
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset
from transformers import AutoTokenizer
from mindformers import MindFormerConfig, TransformerOpParallelConfig
from mindformers import init_context
import json
import logging
from typing import Dict, List
from telechat import TelechatForSequenceClassification
from telechat_config import TelechatConfig
from telechat_transformer import TelechatLoRAModel
import numpy as np
#use mindpet
from mindpet.graph import freeze_delta
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindpet.graph import TrainableParamsCheckPoint
from mindspore import dtype as mstype  # 添加这行导入
# from mindformers.modules.optimizer import FP32StateAdamWeightDecay
from mindformers.core.optim import FP32StateAdamWeightDecay
# from mindpet.utils.config import Config
# 设置环境变量


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义常量
VOCAB_FILE_PATH = "/work/mount/publicModel/TeleChat-12B/telechat_12b_base/"
CHECKPOINT_PATH = "/work/mount/publicModel/TeleChat-12B/telechat_12b_base/telechat_12b_base.ckpt"
YAML_PATH = "/work/home/home/telechat_910B/run_telechat_12b_reward_910b.yaml"



class PreferenceLoss(nn.Cell):
    """奖励模型的偏好学习损失函数"""
    def __init__(self):
        super().__init__()
        self.sigmoid = ops.Sigmoid()
        self.log = ops.Log()
        self.mean = ops.ReduceMean()

    def construct(self, chosen_rewards, rejected_rewards):
        # 确保数据类型一致性
        diff = chosen_rewards - rejected_rewards
        # 添加数值稳定性
        return -self.mean(self.log(self.sigmoid(diff) + 1e-6))

# class RewardTrainingCell(nn.Cell):
#     def __init__(self, model, loss_fn, optimizer):
#         super().__init__(auto_prefix=False)
#         self.model = model
#         self.loss_fn = loss_fn
#         self.optimizer = optimizer
#         self.grad_fn = ops.value_and_grad(self.forward_fn, None, self.optimizer.parameters)
#         self.cast = ops.Cast()  # 添加cast算子
#         self.squeeze = ops.Squeeze(-1)
    
#     def forward_fn(self, input_ids_chosen, attention_mask_chosen, 
#                   input_ids_rejected, attention_mask_rejected):
#         chosen_output = self.model(
#             input_ids=input_ids_chosen,
#             attention_mask=attention_mask_chosen
#         )
#         chosen_rewards = chosen_output["logits"]
        
#         rejected_output = self.model(
#             input_ids=input_ids_rejected,
#             attention_mask=attention_mask_rejected
#         )
#         rejected_rewards = rejected_output["logits"]
#          # 处理奖励值
#         chosen_rewards = self.squeeze(chosen_output["logits"])
#         rejected_rewards = self.squeeze(rejected_output["logits"])#新加的
#         # 使用 F.cast 替代 mstype
#         chosen_rewards = ops.cast(chosen_rewards, ms.float32)
#         rejected_rewards = ops.cast(rejected_rewards, ms.float32)
        
#         loss = self.loss_fn(chosen_rewards, rejected_rewards)
#         return loss
    
#     def construct(self, input_ids_chosen, attention_mask_chosen,
#                  input_ids_rejected, attention_mask_rejected):
#         loss, grads = self.grad_fn(input_ids_chosen, attention_mask_chosen,
#                                  input_ids_rejected, attention_mask_rejected)
#         self.optimizer(grads)
#         return loss
class RewardTrainingCell(nn.Cell):
    def __init__(self, model, loss_fn, optimizer):
        super().__init__(auto_prefix=False)
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_fn = ops.value_and_grad(self.forward_fn, None, self.optimizer.parameters)
        
        # 添加必要的算子
        self.cast = ops.Cast()
        self.squeeze = ops.Squeeze(-1)
        
    def forward_fn(self, input_ids_chosen, attention_mask_chosen, 
                  input_ids_rejected, attention_mask_rejected):
        # 类型转换
        input_ids_chosen = self.cast(input_ids_chosen, ms.int32)
        attention_mask_chosen = self.cast(attention_mask_chosen, ms.int32)
        input_ids_rejected = self.cast(input_ids_rejected, ms.int32)
        attention_mask_rejected = self.cast(attention_mask_rejected, ms.int32)
        
        # 计算奖励
        chosen_output = self.model(
            input_ids=input_ids_chosen,
            attention_mask=attention_mask_chosen
        )
        rejected_output = self.model(
            input_ids=input_ids_rejected,
            attention_mask=attention_mask_rejected
        )
        
        # 处理奖励值
        chosen_rewards = self.squeeze(chosen_output["logits"])
        rejected_rewards = self.squeeze(rejected_output["logits"])
        
        # 确保类型一致
        chosen_rewards = self.cast(chosen_rewards, ms.float32)
        rejected_rewards = self.cast(rejected_rewards, ms.float32)
        
        # 计算损失
        loss = self.loss_fn(chosen_rewards, rejected_rewards)
        return loss
        
    def construct(self, input_ids_chosen, attention_mask_chosen,
                 input_ids_rejected, attention_mask_rejected):
        loss, grads = self.grad_fn(input_ids_chosen, attention_mask_chosen,
                                 input_ids_rejected, attention_mask_rejected)
        self.optimizer(grads)
        return loss
class PreferenceDataset:
    def __init__(self, items, tokenizer, max_length=1024):
        """
        初始化数据集
        Args:
            items: JSON数据列表
            tokenizer: 分词器
            max_length: 序列最大长度
        """
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        """获取单个样本"""
        item = self.items[index]
        
        # 构造输入文本
        chosen_text = f"Question: {item['question']}\nAnswer: {item['chosen']}"
        rejected_text = f"Question: {item['question']}\nAnswer: {item['rejected']}"
        
        # 对文本进行编码
        chosen_tokens = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        rejected_tokens = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        # 提取input_ids和attention_mask
        input_ids_chosen = chosen_tokens['input_ids'][0]
        attention_mask_chosen = chosen_tokens['attention_mask'][0]
        input_ids_rejected = rejected_tokens['input_ids'][0]
        attention_mask_rejected = rejected_tokens['attention_mask'][0]
        
        return (
            input_ids_chosen.astype(np.int32),
            attention_mask_chosen.astype(np.int32),
            input_ids_rejected.astype(np.int32),
            attention_mask_rejected.astype(np.int32)
        )
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.items)


def load_json_data(file_path):
    """加载JSON数据文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
            logger.info(f"Successfully loaded {len(items)} items from {file_path}")
            return items
            
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file: {e}")
        # 打印出错位置附近的内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            error_position = e.pos
            context = content[max(0, error_position-100):min(len(content), error_position+100)]
            logger.error(f"Context around error: {context}")
        return []
def main():
    # 1. 设置运行环境
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
    
    # 2. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(VOCAB_FILE_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 加载模型配置
    config = MindFormerConfig(YAML_PATH)
    # config.use_parallel = False
    print(config.model.model_config.seq_length)
    
    # 4. 创建模型配置
    model_config = TelechatConfig(**config.model.model_config)

    model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    model_config.num_labels = 1
    model_config.hidden_size = 5120  # 确保设置了hidden_size
    model_config.use_past = False  # 奖励模型不需要增量推理
    model_config.is_dynamic = False  # 使用固定序列长度
    model_config.use_flash_attention = False  # 暂时关闭以确保稳定性
    # model_config.seq_length = 1024
    logger.info(f"model_config type: {type(model_config)}")
    logger.info(model_config)
    # 创建模型
    model = TelechatForSequenceClassification(model_config)
    
    # 加载预训练权重
    param_dict = ms.load_checkpoint(CHECKPOINT_PATH)
    new_param_dict = {}
    
    # 只加载transformer部分的权重
    for key, value in param_dict.items():
        if any(key.startswith(prefix) for prefix in model.get_transformer_prefix()):
            new_param_dict[key] = value
    
    # 加载权重
    param_not_load = ms.load_param_into_net(model.transformer, new_param_dict)
    if param_not_load:
        logger.warning(f"Params not loaded: {param_not_load}")
    
    # 7. 应用LoRA
    lora_config = {
    "delta_type": "lora",
    "model_type": "llama",  # 因为是基于llama架构
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": [
        "wq",
        "wk_v", 
        "wo"
    ]
    }

# 创建LoRA delta模型
    # model = lora(
    #     base_model=model,  # 原始模型
    #     config=lora_config
    # )

    # model = TelechatLoRAModel(
    #     base_model=model,   
    #     r=8,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     # target_modules=["wq","wk_v","wo"]
    #     target_modules=["wq","wk_v","wo"]
    # )
    
    # 打印最终模型结构
    def print_model_structure(model, prefix=''):
        for name, param in model.parameters_and_names():
            logger.info(f"{prefix}{name}: {param.shape}")
    
    logger.info("Final model structure:")
    print_model_structure(model)
    
    
    # 8. 准备数据集
    logger.info("Preparing dataset...")
    data_path = "/work/home/home/trian_ppo-main/train_ppo/data/0000 (1).json"
    items = load_json_data(data_path)
    
    if not items:
        logger.error("No valid data loaded!")
        return
        
    # 创建数据集
    dataset = PreferenceDataset(items, tokenizer)
    ds = GeneratorDataset(
        dataset,
        column_names=["input_ids_chosen", "attention_mask_chosen",
                     "input_ids_rejected", "attention_mask_rejected"],
        shuffle=True
    )
    
    # 设置batch size
    ds = ds.batch(batch_size=1)
    
    # 打印数据集信息
    logger.info(f"Dataset created with {len(items)} samples")
    
    # 可选：打印第一个batch的形状
    for batch in ds.create_tuple_iterator():
        input_ids_chosen, attention_mask_chosen, input_ids_rejected, attention_mask_rejected = batch
        logger.info(f"Batch shapes:")
        logger.info(f"input_ids_chosen: {input_ids_chosen.shape}")
        logger.info(f"attention_mask_chosen: {attention_mask_chosen.shape}")
        logger.info(f"input_ids_rejected: {input_ids_rejected.shape}")
        logger.info(f"attention_mask_rejected: {attention_mask_rejected.shape}")
        break
    
    # freeze_delta(
    #     model=model, 
    #     mode='lora',
    #     exclude=['*reward_head*', '*classifier*']  # 排除奖励模型的头部和分类器
    # )
    
    # 10. 创建优化器和损失函数
    optimizer = FP32StateAdamWeightDecay(
        model.trainable_params(),
        learning_rate=config.optimizer.learning_rate,
        beta1=config.optimizer.beta1,
        beta2=config.optimizer.beta2,
        eps=config.optimizer.eps
    )
    loss_fn = PreferenceLoss()
    
    # 11. 创建训练网络
    train_net = RewardTrainingCell(model, loss_fn, optimizer)
    
    # 12. 设置检查点配置，使用TrainableParamsCheckPoint
    config_ck = CheckpointConfig(
        save_checkpoint_steps=1,  # 每个epoch保存一次
        keep_checkpoint_max=10
    )
    checkpoint_dir = "/work/home/home/telechat_910B/checkpoint_lora"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    ckpt_callback = TrainableParamsCheckPoint(
        directory=checkpoint_dir,
        config=config_ck,
        prefix="telechat_reward_lora"
    )
    
    # 13. 训练循环
    epochs = 10
    steps_per_epoch = ds.get_dataset_size() 
    for epoch in range(epochs):
        model.set_train()
        epoch_loss = 0
        for step, batch in enumerate(ds.create_dict_iterator()):
            loss = train_net(
                batch["input_ids_chosen"],
                batch["attention_mask_chosen"],
                batch["input_ids_rejected"],
                batch["attention_mask_rejected"]
            )
            epoch_loss += loss
            
            if step % 100 == 0:
                logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss}")
        
        # 每个epoch结束后保存检查点
        ckpt_callback.step_end(model)
        
        avg_loss = epoch_loss / steps_per_epoch
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss}")
    
    # 14. 保存最终模型
    ckpt_callback.step_end(model, is_last_step=True)
    logger.info("Training completed and model saved!")

if __name__ == "__main__":
    main()