import json
import traceback
import time
import datetime
import uvicorn
import os
import sys
#from loguru import logger
import torch
import gc

# workers = torch.cuda.device_count()
sys.path.insert (0, os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
sys.path.insert (0, os.path.dirname (os.path.dirname (os.path.dirname (os.path.abspath (__file__)))))
sys.path.insert (0, os.path.dirname (os.path.abspath (__file__)))
#from core import workers, gpu_number
from fastapi.middleware.cors import CORSMiddleware
import re
print (os.getcwd ())
from typing import Optional
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, Response, Header, WebSocket, APIRouter, WebSocketDisconnect
# 初始化

from fastapi.responses import StreamingResponse
os.environ["DEVICE_ID"] = "1"
from transformers import AutoTokenizer
from telechat_config import TelechatConfig
from telechat import TelechatForCausalLM
from mindformers import MindFormerConfig, TransformerOpParallelConfig
from mindformers import init_context
from mindformers.tools.utils import str2bool
from mindformers.generation import GenerationConfig
VOCAB_FILE_PATH = sys.args[1]
CHECKPOINT_PATH = sys.args[2]
YAML_PATH = sys.args[3]
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
tokenizer = AutoTokenizer.from_pretrained(VOCAB_FILE_PATH,trust_remote_code=True)
    # 加载模型配置
config = MindFormerConfig(YAML_PATH)
config.use_parallel = False

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
#logger.add ("../log/fastapi{time:YYYY-MM-DD}.log", rotation='00:00')

print ("=============AIGC服务启动==========")

app = FastAPI ()
router = APIRouter ()
app.include_router (router)
app.add_middleware (
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def judge_arg(dialogue):
    '''
    判断入参dialogue是否符合规定
    符合返回true  否则返回false
    规则：开始必须是user user和bot需要交互出现 结尾必须的也是user
    '''
    if not isinstance (dialogue, list):
        return False
    if len (dialogue) == 0:
        return False
    first_item = dialogue[0]
    first_role = first_item['role']
    if first_role != 'user':
        return False
    last_item = dialogue[-1]
    last_role = last_item['role']
    if last_role != 'user':
        return False
    if len (dialogue) > 1:
        for i in range (len (dialogue)):
            if i % 2 == 0:
                if dialogue[i]['role'] != 'user':
                    return False
            if i % 2 == 1:
                if dialogue[i]['role'] != 'bot':
                    return False
    return True


def check_ex(do_sample, max_length, top_k, top_p, temperature, repetition_penalty):
    flag = True
    try:
        if do_sample != None and (do_sample not in [True, False] or not isinstance (do_sample, bool)):
            flag = False
        if max_length != None and (not 0 < max_length < 10000 or not isinstance (max_length, int)):
            flag = False
        if top_k != None and (not 0 < top_k < 100 or not isinstance (top_k, int)):
            flag = False
        if top_p != None and (not 0.0 < top_p < 1.0 or not isinstance (top_p, float)):
            flag = False
        if temperature != None and (not 0.0 < temperature < 1.0 or not isinstance (temperature, float)):
            flag = False
        if repetition_penalty != None and (
                not 1.0 < repetition_penalty < 100.0 or not isinstance (repetition_penalty, float)):
            flag = False
        return flag
    except:
        flag = False
        return flag



def streamresponse_v2(tokenizer, query, history, do_sample, max_length, top_k, top_p, temperature, repetition_penalty):
    result_generator = model.chat (tokenizer, query, history=history, generation_config=generate_config, stream=True,
                                   do_sample=do_sample, max_length=max_length, top_k=top_k, temperature=temperature,
                                   repetition_penalty=repetition_penalty, top_p=top_p)
    t_resp = ''
    while 1:
        try:
            char, _ = next (result_generator)
            if _ is None and char is None:
                break
            else:
                t_resp += char

                yield char
        except StopIteration :
            break


def response_data(seqid, code, message, flag, data):
    res_dict = {
        "seqid": seqid,
        "code": code,
        "message": message,
        "flag": flag,
        "data": data
    }
    res = jsonable_encoder (res_dict)
    print ("### 整个接口的返回结果: ", res)
    return res


@app.route ('/')
def hello_world():
    return "Hello World!"

def parse_data(dialog):
    history = dialog[:-1]
    query = dialog[-1].get ("content")
    return history, query

def _gc():
    gc.collect ()
    if torch.cuda.is_available ():
        torch.cuda.empty_cache ()


@app.post ('/telechat/gptDialog/v2')
async def doc_gptDialog_v2(item: dict, Trace_Id: Optional[str] = Header (None)):
    _gc ()
    print ("Trace-Id=", Trace_Id)
#    logger.info ("接口输入参" + str (item))
#    logger.info ("Trace-Id=" + str (Trace_Id))
    session_res = []
    try:
        dialog = item["dialog"]
    except:
        result_info = response_data ("", "10301", "服务必填参数缺失", "0", "执行失败")
        return result_info
    # 开始进行入参检测
    try:
        qa_tag = judge_arg (dialog)
        if not qa_tag:
            result_info = response_data ("", "10303", "dialog对话格式错误", "0", "执行失败")
            return result_info
    except Exception as e:
        result_info = response_data ("", "10303", "dialog对话格式错误", "0", "执行失败")
        return result_info

    do_sample = item.get ("do_sample", False)
    max_length = item.get ("max_length", 2048)
    top_k = item.get ("top_k", 5)
    top_p = item.get ("top_p", 0.85)
    temperature = item.get ("temperature", 0.3)
    repetition_penalty = item.get ("repetition_penalty", 1.01)
    if not check_ex (do_sample, max_length, top_k, top_p, temperature, repetition_penalty):
        result_info = response_data ("", "10305", "请求参数范围错误", "0", "执行失败")
        return result_info
    try:
        history, query = parse_data (dialog)
        headers = {"code": "10000", "message": "ok", "flag": "1"}
        headers['X-Accel-Buffering'] = 'no'
        return StreamingResponse (
            streamresponse_v2 (tokenizer, query, history, do_sample, max_length, top_k, top_p, temperature, repetition_penalty),
            headers=headers,
            media_type="text/html")
    except Exception as e:
#        logger.error (e)
        import traceback
        traceback.print_exc ()
        result_info = response_data ('', "10903", "服务执行失败", "0", "执行失败")
        _gc ()
        return result_info


@app.post ('/telechat/gptDialog/v4')
async def doc_gptDialog_v3(item: dict, Trace_Id: Optional[str] = Header (None)):
    _gc ()
#    logger.info ("接口输入参" + str (item))
#    logger.info ("Trace-Id=" + str (Trace_Id))
    session_res = []
    try:
        dialog = item["dialog"]
    except:
        result_info = response_data ("", "10301", "服务必填参数缺失", "0", "执行失败")
        return result_info
    # 开始进行入参检测
    try:
        qa_tag = judge_arg (dialog)
        if not qa_tag:
            result_info = response_data ("", "10303", "dialog对话格式错误", "0", "执行失败")
            return result_info
    except Exception as e:
        result_info = response_data ("", "10303", "dialog对话格式错误", "0", "执行失败")
        return result_info
    do_sample = item.get ("do_sample", False)
    max_length = item.get ("max_length", 2048)
    top_k = item.get ("top_k", 5)
    top_p = item.get ("top_p", 0.85)
    temperature = item.get ("temperature", 0.3)
    repetition_penalty = item.get ("repetition_penalty", 1.01)
    if not check_ex (do_sample, max_length, top_k, top_p, temperature, repetition_penalty):
        result_info = response_data ("", "10305", "请求参数范围错误", "0", "执行失败")
        return result_info
    try:
        history, query = parse_data (dialog)
        t_resp = model.chat (tokenizer, query, history=history, generation_config=generate_config, stream=False,
                             do_sample=do_sample, max_length=max_length, top_k=top_k, temperature=temperature,
                             repetition_penalty=repetition_penalty, top_p=top_p)
        res_data = {
            'role': "bot",
            'content': t_resp
        }
        result_info = res_data
    except Exception as e:
#        logger.error (e)
        import traceback
        traceback.print_exc ()
        result_info = response_data ('', "10903", "服务执行失败", "0", "执行失败")
        _gc ()
    return result_info


if __name__ == "__main__":
    ip = "0.0.0.0"
    port = 8070
    # uvicorn.run("stream_telechat_fastapi:app", host=ip, port=port, reload=False)
    uvicorn.run (app, host=ip, port=port, reload=False)


