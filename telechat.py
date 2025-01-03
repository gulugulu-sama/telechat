# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Telechat models' APIs."""
import copy
from typing import Optional, Tuple, Union, List, Dict
from threading import Thread


import mindspore.common.dtype as mstype

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import Tensor, nn
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
try:
    # pylint: disable=W0611
    from mindspore.nn.layer.flash_attention import FlashAttention
    FLASHATTENTION_VALID = True
except ImportError:
    FLASHATTENTION_VALID = False

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.models.utils import cell_reuse
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.modules import KVCachePreprocess
from mindformers.modules.layers import Linear
from mindformers.modules.transformer.transformer import LowerTriangularMaskWithDynamic
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.tools.logger import logger
from mindformers.models.llama.llama_layer import LlamaRMSNorm, FreqsMgr
from mindformers.generation import GenerationConfig

# from research.telechat.telechat_config import TelechatConfig
# from research.telechat.telechat_layer import TelechatEmbedding
# from research.telechat.telechat_transformer import TelechatDecodeLayer
# from research.telechat.generation_utils import History,TelechatIterTextStreamer
from telechat_config import TelechatConfig
from telechat_layer import TelechatEmbedding
from telechat_transformer import TelechatDecodeLayer
from generation_utils import History,TelechatIterTextStreamer
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, Normal
__all__ = ['TelechatModel', 'TelechatForCausalLM']


class TelechatPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TelechatConfig
    base_model_prefix = "telechat"


def layer_compute_dtype(layer, layer_id, offset, parallel_config, n_layers, select_recompute=False):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            layer(Cell) - Represents the transformer block
            parallel_config(dict) - Parallel Config
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(Union[int, List[int]]) - Means the layer_index needs a offset, if there are other modules in the net.
            n_layers(int) - The total layers used for the model.
    """
    pp_dis = max(int((n_layers + 1) / parallel_config.pipeline_stage), 1)
    if isinstance(offset, list):
        if len(offset) != parallel_config.pipeline_stage:
            raise ValueError(f"The length of `offset` {len(offset)} do not match "
                             "`pipeline stage` {parallel_config.pipeline_stage}.")
        i = min(layer_id // pp_dis, parallel_config.pipeline_stage - 1)
        offset_layer = offset[i]
    elif isinstance(offset, int):
        offset_layer = offset
    else:
        raise TypeError(f"`offset` must be `int` of list of `int`, but got {type(offset)}.")

    pp_id = min((layer_id + offset_layer) // pp_dis, parallel_config.pipeline_stage - 1)
    layer.pipeline_stage = pp_id

    # Used for optimizer's fusion tag
    dis = max(int((n_layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        layer.set_comm_fusion(2)
    else:
        layer.set_comm_fusion(int((layer_id + offset_layer) / dis) + 1)
    if isinstance(parallel_config.recompute, bool):
        if parallel_config.recompute and not select_recompute:
            layer.recompute()
    else:
        if parallel_config.recompute.recompute and not select_recompute:
            layer.recompute(
                recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)


class TelechatModel(TelechatPreTrainedModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TelechatDecoderLayer`]
    Args:
        config(TelechatConfig): the config of network

    Returns:
            output: Tensor, the output of Telechat decoderlayer
    """

    def __init__(self,
                 config: TelechatConfig = None):
        super().__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        if config.batch_size or config.use_past:
            Validator.check_positive_int(config.batch_size)
        self.dtype = config.compute_dtype
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.n_head = config.num_heads
        self.head_dim = self.hidden_size // self.n_head
        self.pad_token_id = config.pad_token_id
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.is_dynamic = config.is_dynamic
        self.use_kvcache_op = config.use_kvcache_op
        self.is_flexible_shape = config.is_flexible_shape
        self.use_flash_attention = config.use_flash_attention and FLASHATTENTION_VALID
        if self.use_flash_attention:
            logger.info("Enable flash attention.")
        elif config.use_flash_attention:
            logger.info("Current MindSpore do not support flash attention.")
        logger.info(config.seq_length)
        self.shape = P.Shape()
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.gather = P.Gather()
        self.slice = P.StridedSlice()
        self.max_seq_length = config.seq_length

        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=config.seq_length,
                                  max_position_embedding=config.max_position_embedding,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method)
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention)
        self.tok_embeddings = TelechatEmbedding(vocab_table_size=config.vocab_size,
                                                embedding_size=config.hidden_size,
                                                param_init_type=config.param_init_type)
        self.layers = nn.CellList()
        for layer_id in range(config.num_layers):
            layer = TelechatDecodeLayer(config.batch_size,
                                        config.seq_length,
                                        layer_id,
                                        dim=config.hidden_size,
                                        n_heads=config.num_heads,
                                        n_kv_heads=config.n_kv_heads,
                                        hidden_dropout_prob=config.hidden_dropout_prob,
                                        attention_dropout_prob=config.attention_dropout_prob,
                                        intermediate_size=config.intermediate_size,
                                        ffn_dim_multiplier=config.ffn_dim_multiplier,
                                        norm_eps=config.rms_norm_eps,
                                        qkv_has_bias=config.qkv_has_bias,
                                        compute_dtype=config.compute_dtype,
                                        layernorm_compute_dtype=config.layernorm_compute_type,
                                        softmax_compute_dtype=config.softmax_compute_type,
                                        rotary_dtype=config.rotary_dtype,
                                        param_init_type=config.param_init_type,
                                        use_past=config.use_past,
                                        use_flash_attention=config.use_flash_attention,
                                        is_dynamic=config.is_dynamic,
                                        use_kvcache_op=config.use_kvcache_op,
                                        is_flexible_shape=config.is_flexible_shape,
                                        use_rope_slice=config.use_rope_slice,
                                        parallel_config=config.parallel_config)
            layer_compute_dtype(layer, layer_id, config.offset, config.parallel_config,
                                config.num_layers, select_recompute=config.parallel_config.recompute.select_recompute)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type)
        self.kvcache_preprocess = KVCachePreprocess(max_batch_size=config.batch_size,
                                                    max_seq_length=config.seq_length,
                                                    is_dynamic=config.is_dynamic,
                                                    use_kvcache_op=config.use_kvcache_op,
                                                    is_flexible_shape=config.is_flexible_shape)

        dp = config.parallel_config.data_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.tok_embeddings.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.tok_embeddings.set_comm_fusion(2)
                self.norm_out.set_comm_fusion(2)
            else:
                self.tok_embeddings.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
                self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

            self.tok_embeddings.shard(config.parallel_config)
            self.casual_mask.shard(config.parallel_config)
            self.norm_out.shard((dp, 1, 1))

    # pylint: disable=W0613
    def construct(self, tokens: Tensor, batch_valid_length=None, batch_index=None, zactivate_len=None):
        """
        Forward of telechat model.

        Args:
            tokens: the tokenized inputs with datatype int32
            input_position(Tensor): current position, used by model.predict.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.

        Returns:
            output: Tensor, the output of telechat decoderlayer
        """
        # preprocess
        bs, seq_len = self.shape(tokens)
        if not self.use_past:
            freqs_cis = self.freqs_mgr(self.max_seq_length)
            mask = self.casual_mask(tokens) # mask: [bs, seq, seq]
            # mask = self.casual_mask.post_process(mask)
            kvcache_inputs = None
        else:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr(seq_len)
                mask = self.casual_mask(tokens) # mask: [bs, seq, seq]
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)
                if self.is_dynamic and self.is_flexible_shape and not self.use_kvcache_op:
                    mask = self.casual_mask.increment_slice(
                        self.kvcache_preprocess.range,
                        self.kvcache_preprocess.max_cache_length // bs, batch_valid_length,
                        zactivate_len)
                else:
                    mask = self.casual_mask.increment(self.kvcache_preprocess.range, batch_valid_length, zactivate_len)
            # mask = self.casual_mask.post_process(mask)

            kvcache_inputs = self.kvcache_preprocess(bs, batch_valid_length, batch_index, zactivate_len)

        # tokens: [bs, seq/1]
        h, embedding_weight = self.tok_embeddings(tokens)
        h = self.reshape(h, (bs, seq_len, self.hidden_size))
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            h = self.layers[i](h, freqs_cis, mask, kvcache_inputs=kvcache_inputs)
        output = self.norm_out(h)
        return output, embedding_weight

class TelechatHead(nn.Cell):
    """Head for Telechat to get the logits of each token in the vocab."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 compute_dtype="float16",
                 parallel_config=None):
        super(TelechatHead, self).__init__()
        copied_parallel_config = copy.deepcopy(parallel_config)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = compute_dtype
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        dp = copied_parallel_config.data_parallel
        mp = copied_parallel_config.model_parallel
        if parallel_config.vocab_emb_dp or (out_channels % mp != 0):
            self.matmul = P.MatMul(transpose_b=True).shard(((dp, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((dp, 1), (mp, 1)))

    def construct(self, x, embedding_weight=None):
        out_shape = P.Shape()(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        ori_dtype = F.dtype(x)
        weight = self.cast(embedding_weight, self.dtype)
        x = self.cast(x, self.dtype)
        x = self.matmul(x, weight)
        x = self.cast(x, ori_dtype)
        output = self.reshape(x, out_shape)
        return output

class TelechatForCausalLM(TelechatPreTrainedModel):
    r"""
        Provide telechat training loss or logits through network.

        Args:
            config (TelechatConfig): The config of telechat model.

        Returns:
            output: Tensor, the output of telechat decoderlayer
        """

    @cell_reuse
    def __init__(self, config: TelechatConfig = None):
        super(TelechatForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.config = config
        self.model_name = config.model_name
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.logits_slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather(1)
        self.sub_batch_valid_len = P.Sub()
        self.model = TelechatModel(config=config)
        if self.model_name == 'telechat_12b':
            self.lm_head = Linear(in_channels=config.hidden_size,
                                  out_channels=config.vocab_size,
                                  has_bias=False,
                                  compute_dtype=config.compute_dtype,
                                  param_init_type=config.param_init_type,
                                  skip_redistribution=config.is_dynamic,
                                  weight_init="normal") # meta default: xavier_normal
        else:
            self.lm_head = TelechatHead(in_channels=config.hidden_size,
                                        out_channels=config.vocab_size,
                                        compute_dtype=config.compute_dtype,
                                        parallel_config=config.parallel_config)

        mp = config.parallel_config.model_parallel
        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        if vocab_size % mp != 0:
            logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config)
        self.seq_length = config.seq_length

        dp = config.parallel_config.data_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.slice.shard(((dp, 1),))
            self.logits_slice.shard(((dp, 1, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.mul.shard(((dp, 1), (dp, 1)))
            self.add.shard(((dp, 1), ()))
            self.gather.shard(((dp, 1, 1), (dp,)))
            self.sub_batch_valid_len.shard(((1,), ()))
            if self.model_name == 'telechat_12b':
                if config.parallel_config.vocab_emb_dp or (vocab_size % mp != 0):
                    self.lm_head.shard(strategy_matmul=((dp, 1), (1, 1)))
                else:
                    self.lm_head.shard(strategy_matmul=((dp, 1), (mp, 1)))
            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.load_checkpoint(config)

    # pylint: disable=W0613
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": Tensor(input_ids, mstype.int32)
        }

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None):
        r"""
        TelechatForCausalLM forward.

        Args:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            labels(Tensor): the tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            input_position(Tensor): current position, used by model.predict.
            position_ids(Tensor): Reserved param, not used.
            attention_mask(Tensor): Reserved param, not used.
            input_embeds(Tensor): Reserved param, not used.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.

        Returns:
            Tensor: The loss or (logits, tokens, input_mask) of the network.
        """
        bsz, seqlen = self.shape(input_ids)
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)

        tokens = input_ids
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        if not self.is_first_iteration:
            batch_valid_length = self.sub_batch_valid_len(batch_valid_length, 1)
        output, embedding_weight = self.model(tokens, batch_valid_length, batch_index, zactivate_len)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        if self.model_name == 'telechat_12b':
            logits = self.lm_head(output)
        else:
            logits = self.lm_head(output, embedding_weight)
        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is not None:
            input_mask = labels
        labels = input_ids
        if not self.training:
            if not pre_gather:
                logits = self.reshape(logits, (bsz, seqlen, -1))
            logits = self.cast(logits, mstype.float32)
            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)
            return logits, tokens, input_mask
        logits = self.logits_slice(logits, (0, 0, 0), (bsz, seqlen - 1, self.vocab_size), (1, 1, 1))
        labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
        input_mask = self.slice(input_mask, (0, 1), (bsz, seqlen), (1, 1))
        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss

    def chat(self, tokenizer, question: str = '', history: Union[List[Dict], History] = None, stream: bool = False,
             generation_config: Optional[GenerationConfig] = None, **kwargs):
        """
        Args:
            tokenizer:  the tokenizer of  telechat
            question: question which the model reply in this turn
            history: history which will format the input for telechat
            stream: if return the full text at last or yield the text in token
            generation_config:  configuration for generation
            **kwargs: args which will update the generation config or pass to model forward
        """
        generation_config = generation_config or self.generation_config
        if not generation_config:
            logger.error("generation_config is None")
            raise ValueError("generation_config must not be None")
        if not question:
            logger.error("question is empty")
            raise ValueError("question must not be empty")
        if history is None:
            history = []

        # we update and check generate_config here for building inputs.

        generation_config = copy.deepcopy(generation_config)
        user_id = generation_config.user_token_id
        bot_id = generation_config.bot_token_id
        model_kwargs = generation_config.update(**kwargs)

        # transfer to History
        if not isinstance(history, History):
            history = History(tokenizer, history)

        inputs = self.build_inputs_for_chat(tokenizer, question, history, generation_config, user_id, bot_id)
        history.append({"role": "user", "content": question})
        if stream:
            streamer = TelechatIterTextStreamer(tokenizer, history,skip_prompt=True)
            Thread(target=self.generate, kwargs=dict(
                input_ids=inputs, streamer=streamer,
                generation_config=generation_config, **model_kwargs
            )).start()
            return streamer
        else:
            print(inputs,generation_config)
            outputs = self.generate(inputs, generation_config=generation_config, **model_kwargs)
            print(outputs)
            response = tokenizer.decode(outputs[0][len(inputs):-1])
            history.append({"role": "bot", "content": response})
            return response, history

    def build_inputs_for_chat(self, tokenizer, question, history, generation_config, usr_id, bot_id):
        """
        check history and  build inputs here
        """
        # first tokenize question
        q_token = tokenizer(question)
        qa_history = copy.deepcopy(history)

        # get the max length we should build our inputs in
        model_max_length = self.config.seq_length
        build_max_length = max(0, model_max_length - generation_config.max_new_tokens) \
            if generation_config.max_new_tokens else max(0, generation_config.max_length)
        if build_max_length < 3:
            raise ValueError("the model can not meet the  requirements of input length,Please check config")

        # trunc left
        input_tokens = [usr_id] + q_token["input_ids"][-build_max_length + 1:] + [bot_id]
        length = len(input_tokens)

        while len(qa_history) != 0:
            message = qa_history.pop()
            if message["role"] == "user":
                tokens = [usr_id] + message["input_ids"]
            elif message["role"] == "bot":
                tokens = [bot_id] + message["input_ids"] + [generation_config.eos_token_id]
            else:
                tokens = []
            if len(tokens) + length >= build_max_length:
                break
            else:
                input_tokens = tokens + input_tokens

        return input_tokens

# class TelechatForSequenceClassification(nn.Cell):
#     """用于序列分类的Telechat模型（奖励模型）"""
#     def __init__(self, config):
#         super().__init__()
#         self.num_labels = config.num_labels
#         self.transformer = TelechatModel(config)
#         self.hidden_size = config.hidden_size
        
#         # 分类头
#         self.score = nn.Dense(
#             self.hidden_size,
#             self.num_labels,
#             has_bias=False
#         )
        
#         # 初始化
#         self._init_weights()

#     def _init_weights(self):
#         """初始化权重"""
#         init_range = 0.02
#         self.score.weight.set_data(
#             initializer(Normal(init_range),
#                        self.score.weight.shape,
#                        self.score.weight.dtype)
#         )

#     def get_transformer_prefix(self):
#         """获取transformer参数的前缀列表"""
#         return [
#             'transformer.tok_embeddings',
#             'transformer.layers',
#             'transformer.norm_out'
#         ]

#     def construct(self, 
#                 input_ids,
#                 attention_mask=None,
#                 labels=None,
#                 return_dict=True):
#         """前向传播"""
#         # 1. 获取transformer输出
#         # TelechatModel返回 (output, embedding_weight)
#         transformer_outputs, _ = self.transformer(
#             tokens=input_ids,
#             batch_valid_length=None  # 不使用增量推理
#         )
        
#         # 2. 获取序列表示
#         batch_size = input_ids.shape[0]
        
#         if attention_mask is not None:
#             # 创建mask: [batch_size, seq_length, 1]
#             mask = attention_mask.view(batch_size, -1, 1)
#             # 将padding位置设为很小的值
#             masked_hidden_states = transformer_outputs * mask
            
#             # 使用平均池化获取序列表示
#             mask_sum = ops.maximum(mask.sum(axis=1), 1.0)  # 避免除零
#             pooled_hidden_states = masked_hidden_states.sum(axis=1) / mask_sum
#         else:
#             # 如果没有mask，使用所有token的平均值
#             pooled_hidden_states = transformer_outputs.mean(axis=1)
        
#         # 3. 计算分数
#         logits = self.score(pooled_hidden_states)
        
#         # 4. 计算损失
#         loss = None
#         if labels is not None:
#             if self.num_labels == 1:
#                 loss_fct = nn.MSELoss()
#                 loss = loss_fct(
#                     logits.view(-1),
#                     labels.view(-1).astype(logits.dtype)
#                 )
#             else:
#                 loss_fct = nn.CrossEntropyLoss()
#                 loss = loss_fct(
#                     logits.view(-1, self.num_labels),
#                     labels.view(-1)
#                 )
        
#         if not return_dict:
#             return (loss, logits) if loss is not None else logits
            
#         return {
#             'loss': loss,
#             'logits': logits,
#             'hidden_states': transformer_outputs
#         }

#     def get_rewards(self, input_ids, attention_mask=None):
#         """获取奖励分数"""
#         outputs = self.construct(
#             input_ids,
#             attention_mask=attention_mask,
#             return_dict=True
#         )
#         logits = outputs['logits']
#         return ops.squeeze(logits, -1)  # [batch_size]
   
# class TelechatForSequenceClassification(nn.Cell):
#     """用于序列分类的Telechat模型（奖励模型）"""
#     def __init__(self, config: TelechatConfig = None):
#         super().__init__()
#         self.num_labels = config.num_labels
#         self.transformer = TelechatModel(config)
#         self.hidden_size = config.hidden_size
#         logger.info("TelechatForSequenceClassification config:")
#         logger.info(f"seq_length: {config.seq_length}")
    
#         # self.transformer = TelechatModel(config)
#         if not isinstance(config, TelechatConfig):
#             raise TypeError(f"Expected TelechatConfig, got {type(config)}")
#         # 分类头
#         self.score = nn.Dense(
#             self.hidden_size,
#             self.num_labels,
#             has_bias=False
#         )
        
#         # 初始化
#         self._init_weights()

#     def _init_weights(self):
#         """初始化权重"""
#         init_range = 0.02
#         self.score.weight.set_data(
#             initializer(Normal(init_range),
#                        self.score.weight.shape,
#                        self.score.weight.dtype)
#         )

#     def get_transformer_prefix(self):
#         """获取transformer参数的前缀列表"""
#         return [
#             'transformer.tok_embeddings',
#             'transformer.layers',
#             'transformer.norm_out'
#         ]

#     # def construct(self, 
#     #             input_ids,
#     #             attention_mask=None,
#     #             labels=None,
#     #             return_dict=True):
#     #     """前向传播"""
#     #     # 1. 获取transformer输出
#     #     # TelechatModel返回 (output, embedding_weight)
#     #     input_ids = ops.cast(input_ids, mstype.int32)
#     #     if attention_mask is not None:
#     #         attention_mask = ops.cast(attention_mask, mstype.int32)
            
#     #     transformer_outputs, _ = self.transformer(
#     #         tokens=input_ids,
#     #         batch_valid_length=None # 不使用增量推理
            
#     #     )
        
#     #     # 2. 获取序列表示
#     #     batch_size = input_ids.shape[0]
        
#     #     if attention_mask is not None:
#     #         # 创建mask: [batch_size, seq_length, 1]
#     #         mask = attention_mask.view(batch_size, -1, 1)
#     #         # 将padding位置设为很小的值
#     #         masked_hidden_states = transformer_outputs * mask
            
#     #         # 使用平均池化获取序列表示
#     #         mask_sum = ops.maximum(mask.sum(axis=1), 1.0)  # 避免除零
#     #         pooled_hidden_states = masked_hidden_states.sum(axis=1) / mask_sum
#     #     else:
#     #         # 如果没有mask，使用所有token的平均值
#     #         pooled_hidden_states = transformer_outputs.mean(axis=1)
        
#     #     # 3. 计算分数
#     #     logits = self.score(pooled_hidden_states)
        
#     #     # 4. 计算损失
#     #     loss = None
#     #     if labels is not None:
#     #         if self.num_labels == 1:
#     #             loss_fct = nn.MSELoss()
#     #             loss = loss_fct(
#     #                 logits.view(-1),
#     #                 labels.view(-1).astype(logits.dtype)
#     #             )
#     #         else:
#     #             loss_fct = nn.CrossEntropyLoss()
#     #             loss = loss_fct(
#     #                 logits.view(-1, self.num_labels),
#     #                 labels.view(-1)
#     #             )
        
#     #     if not return_dict:
#     #         return (loss, logits) if loss is not None else logits
            
#     #     return {
#     #         'loss': loss,
#     #         'logits': logits,
#     #         'hidden_states': transformer_outputs
#     #     }
#     def construct(self, 
#                 input_ids,
#                 attention_mask=None,
#                 labels=None,
#                 return_dict=True):
#         """前向传播"""
#         # 1. 获取transformer输出
#         # TelechatModel返回 (output, embedding_weight)
#         input_ids = ops.cast(input_ids, mstype.int32)
#         if attention_mask is not None:
#             attention_mask = ops.cast(attention_mask, mstype.int32)
            
#         transformer_outputs, _ = self.transformer(
#             tokens=input_ids,
#             batch_valid_length=None # 不使用增量推理
            
#         )
        
#         # 2. 获取序列表示
#         batch_size = input_ids.shape[0]
        
#         if attention_mask is not None:
#             # 创建mask: [batch_size, seq_length, 1]

#             # mask = attention_mask.view(batch_size, -1, 1)
#             mask = ops.expand_dims(attention_mask, -1)
#             # 将padding位置设为很小的值
#             masked_hidden_states = transformer_outputs * mask
            
#             # 使用平均池化获取序列表示
#             mask_sum = ops.maximum(mask.sum(axis=1), 1.0)  # 避免除零
#             pooled_hidden_states = masked_hidden_states.sum(axis=1) / mask_sum
#         else:
#             # 如果没有mask，使用所有token的平均值
#             pooled_hidden_states = transformer_outputs.mean(axis=1)
        
#         # 3. 计算分数
#         logits = self.score(pooled_hidden_states)
        
#         # 4. 计算损失
#         loss = None
#         if labels is not None:
#             if self.num_labels == 1:
#                 loss_fct = nn.MSELoss()
#                 loss = loss_fct(
#                     logits.view(-1),
#                     labels.view(-1).astype(logits.dtype)
#                 )
#             else:
#                 loss_fct = nn.CrossEntropyLoss()
#                 loss = loss_fct(
#                     logits.view(-1, self.num_labels),
#                     labels.view(-1)
#                 )
        
#         if not return_dict:
#             return (loss, logits) if loss is not None else logits
            
#         return {
#             'loss': loss,
#             'logits': logits,
#             'hidden_states': transformer_outputs
#         }

#     def get_rewards(self, input_ids, attention_mask=None):
#         """获取奖励分数"""
#         outputs = self.construct(
#             input_ids,
#             attention_mask=attention_mask,
#             return_dict=True
#         )
#         logits = outputs['logits']
#         return ops.squeeze(logits, -1)  # [batch_size]   

class TelechatForSequenceClassification(nn.Cell):
    def __init__(self, config: TelechatConfig = None):
        super().__init__(auto_prefix=True)
        
        # 基础配置
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.compute_dtype = config.compute_dtype
        
        # 初始化算子
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.expand_dims = P.ExpandDims()
        self.sum = P.ReduceSum()
        self.mean = P.ReduceMean()
        self.maximum = P.Maximum()
        self.squeeze = P.Squeeze()
        self.reshape = P.Reshape()
        
        # 设置算子属性
        if config.is_dynamic:
            self.expand_dims.add_prim_attr("skip_redistribution", True)
            self.sum.add_prim_attr("skip_redistribution", True)
            
        # 初始化模型组件
        self.transformer = TelechatModel(config)
        self.score = nn.Dense(
            self.hidden_size,
            self.num_labels,
            has_bias=False,
            dtype=self.compute_dtype
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        logger.info("TelechatForSequenceClassification initialized")
        logger.info(f"seq_length: {config.seq_length}")

    def get_transformer_prefix(self):
        """获取transformer参数的前缀列表"""
        return [
            'transformer.tok_embeddings',
            'transformer.layers',
            'transformer.norm_out'
        ]

    def construct(self, 
                input_ids,
                attention_mask=None,
                labels=None,
                return_dict=True):
        """前向传播"""
        # 类型转换
        input_ids = self.cast(input_ids, mstype.int32)
        if attention_mask is not None:
            attention_mask = self.cast(attention_mask, mstype.int32)
            
        # 获取transformer输出
        transformer_outputs, _ = self.transformer(
            tokens=input_ids,
            batch_valid_length=None
        )
        
        # 获取batch_size
        batch_size = self.shape(input_ids)[0]
        
        # 处理attention mask
        if attention_mask is not None:
            mask = self.expand_dims(attention_mask, -1)
            masked_hidden_states = transformer_outputs * mask
            mask_sum = self.maximum(self.sum(mask, 1), 1.0)
            pooled_hidden_states = self.sum(masked_hidden_states, 1) / mask_sum
        else:
            pooled_hidden_states = self.mean(transformer_outputs, axis=1)
        
        # 计算logits
        logits = self.score(pooled_hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = self.mse_loss(
                    self.reshape(logits, (-1,)),
                    self.cast(self.reshape(labels, (-1,)), logits.dtype)
                )
            else:
                loss = self.ce_loss(
                    self.reshape(logits, (-1, self.num_labels)),
                    self.reshape(labels, (-1,))
                )
        
        if not return_dict:
            return (loss, logits) if loss is not None else logits
            
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': transformer_outputs
        }

    def get_rewards(self, input_ids, attention_mask=None):
        """获取奖励分数"""
        outputs = self.construct(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs['logits']
        return self.squeeze(logits, -1)  # [batch_size]
class CriticModel(nn.Cell):
    def __init__(self, config):
        super().__init__()
        # 1. 基础设置
        self.model_type = 'telechat_12b'
        self.phase = "train"  # 默认训练模式
        
        # 2. 模型组件
        self.model = TelechatModel(config)
        self.output_dtype = mstype.float16
        self.sequence_len = config.seq_length
        self.pad_token_id = config.pad_token_id
        
        # 3. 操作算子
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        
        # 4. 价值头
        self.v_head = nn.Dense(
            in_channels=config.hidden_size,
            out_channels=1,
            has_bias=True,
            compute_dtype=config.compute_dtype,
            param_init_type=config.param_init_type
        ).to_float(mstype.float16)
        
        # 5. 并行配置
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.v_head.shard(strategy_matmul=((dp, mp), (1, mp)))
        self.v_head.weight.parallel_optimizer = False
        if config.parallel_config.pipeline_stage > 1:
            self.v_head.pipeline_stage = config.parallel_config.pipeline_stage - 1
            
        self.sigmoid = nn.Sigmoid()

    def set_train(self, mode=True):
        """设置训练/评估模式"""
        super().set_train(mode)
        self.phase = "train" if mode else "eval"
        self.model.set_train(mode)
        return self

    def construct(self, input_ids, attention_mask=None, input_position=None):
        """前向传播"""
        # 1. 获取输入形状
        batch_size, seq_length = self.shape(input_ids)
        
        # 2. 处理输入
        if self.phase == "train":
            # 训练模式：去掉最后一个token
            tokens = self.slice(
                input_ids,
                (0, 0),
                (batch_size, seq_length - 1),
                (1, 1)
            )
            seq_length = seq_length - 1
        else:
            # 推理模式：使用完整序列
            tokens = input_ids
            
        # 3. 创建attention mask（如果没有提供）
        if attention_mask is None:
            attention_mask = self.cast(
                self.not_equal(tokens, self.pad_token_id),
                mstype.float32
            )
            
        # 4. 获取模型输出
        output_states, _ = self.model(
            tokens,
            attention_mask=attention_mask,
            input_position=input_position
        )
        
        # 5. 计算价值
        values = self.v_head(output_states)
        values = self.reshape(values, (batch_size, -1))  # -1 自动适应seq_length
        values = self.sigmoid(values)
        
        return values

    def get_value(self, input_ids, attention_mask=None):
        """获取价值评估（用于推理）"""
        self.set_train(False)
        values = self.construct(input_ids, attention_mask)
        return values
   

   
class CausalLMHydraWithValueHead(TelechatForCausalLM):
    """在TelechatForCausalLM基础上添加价值头的模型"""
    def __init__(self, config: TelechatConfig, ppo_config=None):
        super().__init__(config)
        
        # 1. 初始化价值头
        if self.model_name == 'telechat_12b':
            self.value_head = Linear(
                in_channels=config.hidden_size,
                out_channels=1,
                has_bias=False,
                compute_dtype=config.compute_dtype,
                param_init_type=config.param_init_type,
                skip_redistribution=config.is_dynamic,
                weight_init="normal"
            )
        else:
            self.value_head = TelechatHead(
                in_channels=config.hidden_size,
                out_channels=1,
                compute_dtype=config.compute_dtype,
                parallel_config=config.parallel_config
            )
            
        # 2. 设置并行策略
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            if self.model_name == 'telechat_12b':
                if config.parallel_config.vocab_emb_dp:
                    self.value_head.shard(strategy_matmul=((dp, 1), (1, 1)))
                else:
                    self.value_head.shard(strategy_matmul=((dp, 1), (mp, 1)))
            if config.parallel_config.pipeline_stage > 1:
                self.value_head.pipeline_stage = config.parallel_config.pipeline_stage - 1

        # 3. 初始化必要的操作符
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.gather = P.GatherD()
        
    def construct(self, input_ids, samples=None, attention_mask=None, return_value=False, **kwargs):
        """扩展construct方法以支持价值预测和策略输出
        
        Args:
            input_ids: 输入token ids
            samples: 用于计算log概率的样本
            attention_mask: 注意力掩码
            return_value: 是否返回价值预测
            **kwargs: 其他参数
        """
        # 1. 获取原始输出
        outputs = super().construct(input_ids, attention_mask=attention_mask, **kwargs)
        
        # 2. 如果需要计算策略的log概率
        if samples is not None:
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            # 计算log概率
            log_probs = self.compute_logprobs(logits, samples)
            outputs = log_probs
        
        # 3. 如果需要返回价值
        if return_value:
            # 获取最后一层的隐藏状态
            hidden_states = self.model(input_ids, attention_mask=attention_mask)[0]
            
            # 计算价值
            if self.model_name == 'telechat_12b':
                values = self.value_head(hidden_states)
            else:
                values = self.value_head(hidden_states, self.model.get_embedding_weight())
                
            # 返回结果
            if isinstance(outputs, tuple):
                return outputs + (values,)
            else:
                return outputs, values
                
        return outputs

    def compute_logprobs(self, logits, samples):
        """计算log概率
        
        Args:
            logits: 模型输出的logits
            samples: 用于计算log概率的样本
        Returns:
            log_probs: 计算得到的log概率
        """
        # 1. 计算log softmax
        log_probs = self.log_softmax(logits)
        
        # 2. 获取特定token的log概率
        samples = samples.expand_dims(-1)
        log_probs = self.gather(log_probs, -1, samples)
        
        return log_probs.squeeze(-1)

    def get_value(self, input_ids, attention_mask=None):
        """获取状态价值"""
        hidden_states = self.model(input_ids, attention_mask=attention_mask)[0]
        if self.model_name == 'telechat_12b':
            values = self.value_head(hidden_states)
        else:
            values = self.value_head(hidden_states, self.model.get_embedding_weight())
        return values

    def forward_with_value(self, input_ids, attention_mask=None):
        """同时获取策略输出和价值"""
        return self.construct(input_ids, attention_mask=attention_mask, return_value=True)