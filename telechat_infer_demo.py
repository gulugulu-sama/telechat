import os

# 修改环境变量
os.environ["DEVICE_ID"] = "1"

from transformers import AutoTokenizer
from telechat_config import TelechatConfig
from telechat import TelechatForCausalLM
from mindformers import MindFormerConfig, TransformerOpParallelConfig
from mindformers import init_context
from mindformers.tools.utils import str2bool
from mindformers.generation import GenerationConfig

VOCAB_FILE_PATH = "/work/mount/publicModel/TeleChat-12B/telechat_12b_base/"
CHECKPOINT_PATH = "/work/mount/publicModel/TeleChat-12B/telechat_12b_base/telechat_12b_base.ckpt"
YAML_PATH = "telechat-service/910B/telechat_910B/run_telechat_12b_predict_910b.yaml"
INFER_DICT = {
    "max_length": 512,
    "do_sample": False,
    "use_past": True,
    "temperature": 0.3,
    "top_k": 5,
    "top_p": 0.85,
    "repetition_penalty": 1.0,
    "pad_token_id": 3,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "user_token_id": 20,
    "bot_token_id": 21
}
def main():
    # 加载tokenizer相关
    tokenizer = AutoTokenizer.from_pretrained(VOCAB_FILE_PATH,trust_remote_code=True)
    # 加载模型配置
    config = MindFormerConfig(YAML_PATH)
    config.use_parallel = False

    # 初始化环境
    init_context(context_config=config.context)

    model_config = TelechatConfig(**config.model.model_config)
    model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    model_config.batch_size = 1
    model_config.use_past = True
    model_config.use_flash_attention = False

    if CHECKPOINT_PATH and not config.use_parallel:
        model_config.checkpoint_name_or_path = CHECKPOINT_PATH
    print(f"config is: {model_config}")

    # 加载模型
    model = TelechatForCausalLM(model_config)

    # 推理参数
    generate_config = GenerationConfig.from_dict(INFER_DICT)

    
    #  chat(bot)模型多轮演示
    print("*" * 10 + "多轮输入演示" + "*" * 10)
    question = "你是谁？"
    print("提问:", question)
    answer, history = model.chat(tokenizer=tokenizer, question=question, history=[], generation_config=generate_config,
                                 stream=False,max_length=2048,do_sample=False,temperature=0.3,top_k=5,
                                 top_p=0.85,repetition_penalty=1.01)
    print("回答:", answer)
    print("截至目前的聊天记录是:", history)

    question = "你是谁训练的"
    print("提问:", question)
    # 将history传入
    answer, history = model.chat(tokenizer, question=question, history=history, generation_config=generate_config,
                                 stream=False)
    print("回答是:", answer)
    print("截至目前的聊天记录是:", history)

    # 也可以这么调用传入history
    history = [
        {"role": "user", "content": "你是谁"},
        {"role": "bot", "content": "我是telechat"},
    ]

    question = "你是谁训练的"
    print("提问:", question)
    answer, history = model.chat(tokenizer, question=question, history=history, generation_config=generate_config,
                                 stream=False)
    print("回答是:", answer)
    print("截至目前的聊天记录是:", history)

    
    # chat(bot)模型 流式返回演示
    print("*" * 10 + "流式输入演示" + "*" * 10)
    question = "你是谁？"
    print("提问:", question)
    gen = model.chat(tokenizer, question=question, history=[], generation_config=generate_config,
                     stream=True)
    for answer, history in gen:
        print("回答是:", answer)
        print("截至目前的聊天记录是:", history)



if __name__ == '__main__':
    main()
