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
"""Telechat transformer Layer's APIs."""
from typing import Tuple, Optional
import math

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator

from mindspore import nn, __version__
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
try:
    from mindspore.nn.layer.flash_attention import FlashAttention
    FLASHATTENTION_VALID = True
except ImportError:
    FLASHATTENTION_VALID = False

from mindformers.models.llama.llama_layer import LlamaRMSNorm, LlamaRotaryEmbedding
from mindformers.modules.layers import _check_input_dtype, Dropout
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules import KVCacheMgr

from mindformers.tools.utils import is_version_ge
from mindformers.tools.logger import logger
# from research.telechat.telechat_layer import TelechatLinear, TelechatFeedForward
from telechat_layer import TelechatLinear, TelechatFeedForward
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn import Cell

class TelechatAttention(nn.Cell):
    r"""
    This is an implementation of multihead attention in Telechat.

    Args:
            - **batch_size** (int): The batch size of the input tensor when do increnmental prediction. Should be a
                positive value.
                When do training or prediction, the argument will not work and the user can just pass None to the
                argument.
            - **src_seq_length** (int): The sequence length of the query vector.
            - **tgt_seq_length** (int): The sequence length of the key and value vector.
            - **dim** (int): The hidden size of the input.
            - **head_dim** (int): The dim of head.
            - **n_heads** (int): The number of the heads.
            - **compute_dtype** (dtype.Number): The computation type of dense. Default mstype.float16.
                Should be mstype.float32 or mstype.float16.
            - **softmax_compute_type** (dtype.Number): The type of softmax computation module. Default mstype.float32.
                Should be mstype.float32 or mstype.float16.
            - **param_init_type** (dtype.Number): The parameter initialization type of the module. Default mstype.
                float32. Should be mstype.float32 or mstype.float16.
            - **qkv_has_bias** (bool): Whether Q/K/V in attention has bias or not.
            - **use_past** (bool): Use the past state to compute, used for incremental prediction.
                For example, if we have two words and want to generate the ten more words.
                We just need to compute the two words' state only once, and generate the next word one by one.
                When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`. At this moment,
                pass the single step's input tensor, and loop it. Default False.
            - **parallel_config** (OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

    Inputs:
            - **x** (Tensor) - The input tokens with shape (batch_size, src_seq_length, hidden_size) or
                (batch_size * src_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
                Otherwise, must be (batch_size, 1, hidden_size)
            - **freqs_cis** (Tuple) - The precompute freqs and mask for rotary position embedding used in attention.
            - **attention_mask** (Tensor) - If the use_past is False or is_first_iteration=True, the attention mask
                matrix should ba (batch_size, src_seq_length, tgt_seq_length), or None. None means there will be no mask
                in softmax computation. Otherwise, the mask must be (batch_size, 1, tgt_seq_length)
            - **key_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, head_dim, tgt_seq_length).
                The past calculated key vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **value_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, tgt_seq_length,
                head_dim).
                The past calculated value vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape (batch_size,) the past calculated the index.
                Used for incremental prediction when the use_past is True. Default None.

    Outputs:
            Tuple, a tuple contains(`output`, `layer_present`)

            - **output** (Tensor) - Tensor, the float tensor of the output of the layer with
                shape (batch_size, src_seq_length, hidden_size) or (batch_size * src_seq_length, hidden_size),
                if the use_past is False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size).

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
                ((batch_size, num_heads, head_dim, tgt_seq_length),
                (batch_size, num_heads, tgt_seq_length, head_dim)).
    """
    def __init__(self,
                 batch_size,
                 seq_length,
                 dim: int = 512,
                 n_heads: int = 8,
                 hidden_dropout_prob: float = 1.0,
                 attention_dropout_prob: float = 1.0,
                 n_kv_heads: Optional[int] = None,
                 compute_dtype=mstype.float16,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 qkv_has_bias=False,
                 use_past=False,
                 is_dynamic=False,
                 use_kvcache_op=False,
                 is_flexible_shape=False,
                 use_rope_slice=False,
                 use_flash_attention=False,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_size = dim
        self.n_head = n_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_head // self.n_kv_head
        self.kv_dim = self.n_kv_head * self.head_dim

        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past
        self.use_flash_attention = use_flash_attention and FLASHATTENTION_VALID

        if self.hidden_size % self.n_head != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'n_head', but got the hidden_size is {} and the n_head is {}."
                             .format(self.hidden_size, self.n_head))
        if self.n_kv_head % parallel_config.model_parallel != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'n_kv_head' must be a multiple of "
                             "'parallel_config.model_parallel', but got the n_kv_head is {} "
                             "and the parallel_config.model_parallel  is {}."
                             .format(self.n_kv_head, parallel_config.model_parallel))

        self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)

        self.shape = P.Shape()
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.transpose = P.Transpose()
        self.merger_head_transpose = P.Transpose()
        self.batch_matmul = P.BatchMatMul()
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.mul = P.Mul()
        self.add = P.Add()
        self.softmax = P.Softmax()
        self.cast = P.Cast()
        self.cast_attn = P.Cast()
        self.tile_kv = P.Tile()
        self.split = P.Split(output_num=2, axis=-1)
        self.apply_rotary_emb = LlamaRotaryEmbedding(self.head_dim, rotary_dtype, use_rope_slice=use_rope_slice)
        self.attention_dropout = Dropout(1-self.attention_dropout_prob)
        self.wo = TelechatLinear(in_channels=self.hidden_size,
                                 out_channels=self.hidden_size,
                                 has_bias=True,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type,
                                 skip_redistribution=is_dynamic,
                                 keep_prob=1-self.hidden_dropout_prob)
        self.wq = TelechatLinear(self.hidden_size,
                                 self.hidden_size,
                                 has_bias=qkv_has_bias,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type,
                                 skip_redistribution=is_dynamic)
        self.wk_v = TelechatLinear(self.hidden_size,
                                   self.n_kv_head * self.head_dim * 2,
                                   has_bias=qkv_has_bias,
                                   compute_dtype=compute_dtype,
                                   param_init_type=param_init_type,
                                   skip_redistribution=is_dynamic)
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.transpose.shard(((dp, 1, mp, 1),))
            self.merger_head_transpose.shard(((dp, mp, 1, 1),))
            self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.mul.shard(((dp, mp, 1, 1), ()))
            self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
            self.softmax.shard(((dp, mp, 1, 1),))
            self.tile_kv.shard(((dp, mp, 1, 1),))

            self.apply_rotary_emb.shard((dp, mp, 1, 1))

            if qkv_has_bias:
                self.wq.shard(((dp, 1), (mp, 1)), ((dp, mp), (mp,)))
                self.wk_v.shard(((dp, 1), (mp, 1)), ((dp, mp), (mp,)))
            else:
                self.wq.shard(((dp, 1), (mp, 1)))
                self.wk_v.shard(((dp, 1), (mp, 1)))
            self.wo.shard(((dp, mp), (1, mp)))
            if parallel_config.use_seq_parallel and self.is_first_iteration:
                self.wo.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))
            if parallel_config.recompute.select_recompute:
                self.apply_rotary_emb.recompute()
                self.tile_kv.recompute()
                self.batch_matmul_q_k.recompute()
                self.mul.recompute()
                self.add.recompute()
                self.cast_attn.recompute()
                self.softmax.recompute()
                self.batch_matmul.recompute()

        if not is_version_ge(__version__, "2.2.0"):
            self.use_flash_attention = False
            logger.info("Current MindSpore do not support flash attention, please upgrade to 2.2.0 or higher")
        if self.use_flash_attention:
            self.flash_attention = FlashAttention(self.head_dim, n_heads, dp=dp, mp=mp, next_block_num=0,
                                                  high_precision=True)

        if self.use_past:
            self.kvcache_mgr = KVCacheMgr(self.n_kv_head, self.head_dim,
                                          max_batch_size=batch_size,
                                          max_seq_length=seq_length,
                                          compute_dtype=compute_dtype,
                                          is_dynamic=is_dynamic,
                                          use_kvcache_op=use_kvcache_op,
                                          is_flexible_shape=is_flexible_shape)
            self.kvcache_mgr.shard(parallel_config)

    def construct(self, x: Tensor, freqs_cis: Tuple[Tensor, Tensor], mask=None, kvcache_inputs=None):
        """Forward process of the MultiHeadAttention"""
        ori_dtype = x.dtype
        # [bs, seq/1, hidden_dim]
        bs, seq_len, _ = self.shape(x)
        # [bs * seq/1, hidden_dim]
        query = self.cast(self.wq(x), self.dtype)  # dp, 1 -> dp, mp
        key_value = self.cast(self.wk_v(x), self.dtype)  # dp, 1 -> dp, mp

        key_value = self.reshape(key_value, (seq_len, -1, self.n_kv_head, self.head_dim * 2))
        key, value = self.split(key_value)

        if self.use_past and not self.is_first_iteration:
            query = self.reshape(query, (bs, self.n_head, 1, self.head_dim))
            key = self.reshape(key, (bs, self.n_kv_head, 1, self.head_dim))
            value = self.reshape(value, (bs, self.n_kv_head, 1, self.head_dim))
        else:
            query = self.reshape(query, (bs, seq_len, self.n_head, self.head_dim))
            key = self.reshape(key, (bs, seq_len, self.n_kv_head, self.head_dim))
            value = self.reshape(value, (bs, seq_len, self.n_kv_head, self.head_dim))
            # [bs, seq/1, n_head/n_kv_head, head_dim]
            query = self.transpose(query, (0, 2, 1, 3))
            key = self.transpose(key, (0, 2, 1, 3))
            value = self.transpose(value, (0, 2, 1, 3))
        # [bs, n_head/n_kv_head, seq/1, head_dim]
        query, key = self.apply_rotary_emb(query, key, freqs_cis) # dp, mp, 1, 1
        # kv cache: [bs, n_kv_head, 1, head_dim] -> [bs, n_kv_head, seq, head_dim]
        if self.use_past:
            key, value = self.kvcache_mgr(key, value, kvcache_inputs)
        # kv share: [bs, n_kv_head, seq, head_dim] -> [bs, n_head, seq, head_dim]
        key = self._repeat_kv(key, self.n_rep)
        value = self._repeat_kv(value, self.n_rep)
        # q, k, v: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim], [bs, n_head, seq, head_dim]
        if self.use_flash_attention:
            attention = self.flash_attention(query, key, value, mask)
            attention = self._merge_heads(attention)
        else:
            attention = self._attn(query, key, value, mask)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        output = self.wo(attention) # dp, mp -> dp, 1 / dp * mp, 1
        output = self.cast(output, ori_dtype)

        return output

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = self.shape(x)
        x = self.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile_kv(x, (1, 1, rep, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d or 3d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        # [bs, n_head, seq/1, head_dim]
        x = self.merger_head_transpose(x, (0, 2, 1, 3)) # dp,mp,1,1 -> dp,1,mp,1
        # [bs, seq/1, n_head, head_dim]
        bs, seq_len, n_head, head_dim = self.shape(x)
        # [bs, seq/1, hidden_dim]
        new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            mask: the attention mask adder matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # q, k: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim]
        score = self.batch_matmul_q_k(query, key)
        # score: [bs, n_head, seq/1, seq]
        score = self.mul(score, self.inv_norm_factor)
        score = self.add(mask, score)

        attention_probs = self.softmax(self.cast_attn(score, self.softmax_dtype))
        # score, v: [bs, n_head, seq/1, seq], [bs, n_head, seq, head_dim]
        attention_probs = self.attention_dropout(attention_probs)
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.dtype), value)
        # [bs, n_head, seq/1, head_dim]
        attention_merge = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return attention_merge


class TelechatDecodeLayer(nn.Cell):
    r"""
        Transformer Layer. This is an implementation of the single layer of the transformer
        encoder layer, including multihead attention and feedward layer.

        Args:
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            seq_length(int): The input sequence length.
            layer_id(int): The layer id of current transformer block layer.
            dim(int): The hidden size of the input.
            num_heads(int): The number of the heads.
            norm_eps (float): The epsilon value of the denominator. Default 1e-5.
            compute_dtype(dtype.Number): The computation type of the layer.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            layernorm_compute_type(dtype.Number): The computation type of the norm.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            softmax_compute_type(dtype.Number): The computation type of the softmax in the attention.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            param_init_type(dtype.Number): The parameter initialization type of the module.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            qkv_has_bias(bool): Whether Q/K/V in attention has bias or not.
            use_past(bool): Use the past state to compute, used for incremental prediction. For example, if we have two
                words and want to generate the ten more words. We just need to compute the two words' state only once,
                and generate the next word one by one. When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`.
                At this moment, pass the single step's input tensor, and loop it. Default False.
            parallel_config(OpParallelConfig, MoEParallelConfig): The parallel configure. When MoE is applied,
                MoEParallelConfig is effective, otherwise OpParallelConfig is effective. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

        Inputs:
            - **x** (Tensor) - Float Tensor, shape should be [batch_size, seq_length, hidden_size] or
              [batch_size * seq_length, hidden_size], if the use_past is False or is_first_iteration=True. Otherwise,
              should be [batch_size, 1, hidden_size]
            - **freqs_cis** (Tuple) - The precompute freqs and mask for rotary position embedding used in attention.
            - **input_mask** (Tensor) - Float Tensor, If the use_past is False or is_first_iteration=True,
              the attention mask matrix should ba [batch_size, seq_length, seq_length], or None. None means there will
              be no mask in softmax computation. Otherwise, should be [batch_size, 1, hidden_size]
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`).

            - **output** (Tensor) - The float tensor of the output of the layer with
              shape (batch_size, seq_length, hidden_size) or (batch_size * seq_length, hidden_size), if the use_past is
              False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size)

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
              ((batch_size, num_heads, head_dim, seq_length),
              (batch_size, num_heads, seq_length, head_dim)).

    """
    def __init__(self,
                 batch_size,
                 seq_length,
                 layer_id,
                 dim: int = 512,
                 n_heads: int = 8,
                 hidden_dropout_prob: float = 1.0,
                 attention_dropout_prob: float = 1.0,
                 n_kv_heads: Optional[int] = None,
                 intermediate_size: Optional[int] = None,
                 ffn_dim_multiplier: Optional[int] = None,
                 norm_eps: float = 1e-5,
                 compute_dtype=mstype.float16,
                 layernorm_compute_dtype=mstype.float32,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 qkv_has_bias=False,
                 use_past=False,
                 is_dynamic=False,
                 use_kvcache_op=False,
                 is_flexible_shape=False,
                 use_rope_slice=False,
                 use_flash_attention=False,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
        if batch_size or use_past:
            Validator.check_positive_int(batch_size)
        self.batch_size = batch_size

        self.seq_length = seq_length
        self.layer_id = layer_id
        self.hidden_size = dim
        self.n_head = n_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.head_dim = self.hidden_size // self.n_head
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.dtype = compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past

        self.shape = P.Shape()
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.add = P.Add()
        self.attention_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.ffn_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.attention = TelechatAttention(batch_size=batch_size,
                                           seq_length=seq_length,
                                           dim=dim,
                                           n_heads=n_heads,
                                           hidden_dropout_prob=hidden_dropout_prob,
                                           attention_dropout_prob=attention_dropout_prob,
                                           n_kv_heads=n_kv_heads,
                                           compute_dtype=compute_dtype,
                                           softmax_compute_dtype=softmax_compute_dtype,
                                           rotary_dtype=rotary_dtype,
                                           param_init_type=param_init_type,
                                           qkv_has_bias=qkv_has_bias,
                                           use_past=use_past,
                                           is_dynamic=is_dynamic,
                                           use_kvcache_op=use_kvcache_op,
                                           is_flexible_shape=is_flexible_shape,
                                           use_rope_slice=use_rope_slice,
                                           use_flash_attention=use_flash_attention,
                                           parallel_config=parallel_config)
        self.feed_forward = TelechatFeedForward(dim=self.hidden_size,
                                                intermediate_size=intermediate_size,
                                                hidden_dim=4 * self.hidden_size,
                                                hidden_dropout_prob=hidden_dropout_prob,
                                                ffn_dim_multiplier=ffn_dim_multiplier,
                                                compute_dtype=compute_dtype,
                                                param_init_type=param_init_type,
                                                is_dynamic=is_dynamic)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.feed_forward.shard(parallel_config)
            self.add.shard(((dp, 1, 1), (dp, 1, 1)))
            self.attention_norm.shard((dp, 1, 1))
            self.ffn_norm.shard((dp, 1, 1))
            self.feed_forward.mul.shard(((dp, 1, mp), (dp, 1, mp)))

        if parallel_config.use_seq_parallel and self.is_first_iteration:
            self.add.shard(((dp, mp, 1), (dp, mp, 1)))
            self.attention_norm.shard((dp, mp, 1))
            self.ffn_norm.shard((dp, mp, 1))
            self.feed_forward.w2.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))

    def construct(self, x, freqs_cis, mask=None, kvcache_inputs=None):
        """ Forward of transformer block. """
        self._check_input(x, freqs_cis, mask)
        # [bs, seq/1, hidden_dim]
        input_x = self.attention_norm(x)
        # [bs, seq/1, hidden_dim]
        h = self.attention(input_x, freqs_cis, mask, kvcache_inputs)
        h = self.add(x, h)
        ffn_norm = self.ffn_norm(h)
        # [bs, seq/1, hidden_dim]
        ffn_out = self.feed_forward(ffn_norm)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        out = self.add(h, ffn_out)
        return out

    def _check_input(self, x, freqs_cis, mask):
        r"""Check inputs"""
        _check_input_dtype(
            x.dtype, "x", [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        freqs_cos, freqs_sin, swap_mask = freqs_cis
        _check_input_dtype(freqs_cos.dtype, "freqs_cos",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        _check_input_dtype(freqs_sin.dtype, "freqs_sin",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        if swap_mask is not None:
            _check_input_dtype(swap_mask.dtype, "swap_mask",
                               [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        if mask is not None:
            _check_input_dtype(mask.dtype, "input_mask",
                               [mstype.float32, mstype.float16, mstype.bfloat16, mstype.uint8], self.cls_name)
        return True
# from mindspore import scope





# class LoRALinear(Cell):
#     """LoRA包装类，用于线性层"""
#     def __init__(self, 
#                  linear: nn.Dense,  # 原始线性层
#                  r: int = 8,        # LoRA秩
#                  lora_alpha: int = 16,
#                  lora_dropout: float = 0.05,
#                  merge_weights: bool = False,
#                  layer_id: str = None):  # 添加layer_id参数
#         super().__init__()
        
#         self.layer_id = layer_id or str(id(self))
        
#         # 创建新的linear层
#         self.linear = nn.Dense(
#             linear.in_channels,
#             linear.out_channels,
#             has_bias=linear.has_bias,
#             dtype=linear.dtype
#         )
        
#         # 复制原始权重
#         self.linear.weight.set_data(linear.weight)
#         if linear.has_bias and hasattr(linear, 'bias') and linear.bias is not None:
#             self.linear.bias.set_data(linear.bias)
        
#         # 重命名linear层参数
#         if hasattr(self.linear.weight, 'name'):
#             self.linear.weight.name = f'{self.layer_id}.linear.weight'
#         if hasattr(self.linear, 'bias') and self.linear.bias is not None and hasattr(self.linear.bias, 'name'):
#             self.linear.bias.name = f'{self.layer_id}.linear.bias'
        
#         # LoRA矩阵
#         self.lora_A = nn.Dense(
#             linear.in_channels, 
#             r,
#             has_bias=False,
#             dtype=linear.dtype
#         )
#         self.lora_B = nn.Dense(
#             r,
#             linear.out_channels,
#             has_bias=False,
#             dtype=linear.dtype
#         )
        
#         # 重命名LoRA参数
#         if hasattr(self.lora_A.weight, 'name'):
#             self.lora_A.weight.name = f'{self.layer_id}.lora_A.weight'
#         if hasattr(self.lora_B.weight, 'name'):
#             self.lora_B.weight.name = f'{self.layer_id}.lora_B.weight'
        
#         self.r = r
#         self.lora_alpha = lora_alpha
#         self.lora_dropout = nn.Dropout(p=lora_dropout)
#         self.scaling = self.lora_alpha / self.r
#         self.merge_weights = merge_weights
        
#         if merge_weights:
#             self.merge()

#     def construct(self, x):
#         # 原始前向传播
#         base_out = self.linear(x)
        
#         # LoRA前向传播
#         lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        
#         return base_out + lora_out
        
#     def merge(self):
#         """合并LoRA权重到原始权重"""
#         if self.merge_weights:
#             self.linear.weight.set_data(
#                 self.linear.weight + 
#                 self.lora_B.weight @ self.lora_A.weight * self.scaling
#             )
#             self.lora_A.weight.set_data(
#                 ops.zeros_like(self.lora_A.weight)
#             )
#             self.lora_B.weight.set_data(
#                 ops.zeros_like(self.lora_B.weight)
#             )

# class TelechatLoRAModel(nn.Cell):
#     """LoRA包装类，用于整个模型"""
#     def __init__(self,
#                  base_model: TelechatDecodeLayer,
#                  r: int = 8,
#                  lora_alpha: int = 16,
#                  lora_dropout: float = 0.05,
#                  target_modules: list = ["wq", "wk_v", "wo"]):
#         super().__init__()
#         self.base_model = base_model
        
#         # 替换目标模块为LoRA版本
#         self._replace_modules(
#             self.base_model,
#             target_modules,
#             r,
#             lora_alpha,
#             lora_dropout,
#             prefix=""
#         )
        
#         # 验证参数名称唯一性
#         self._verify_param_names()

#     def _replace_modules(self, model, target_modules, r, lora_alpha, lora_dropout, prefix, block_idx=0):
#         """递归替换目标模块"""
#         for idx, (name, module) in enumerate(model.name_cells().items()):
#             current_prefix = f"{prefix}/{name}/{idx}" if prefix else f"{name}/{idx}"
            
#             if isinstance(module, TelechatAttention):
#                 if hasattr(module, 'wq'):
#                     module.wq = LoRALinear(
#                         module.wq,
#                         r=r,
#                         lora_alpha=lora_alpha,
#                         lora_dropout=lora_dropout,
#                         layer_id=f'{current_prefix}/wq'
#                     )
#                 if hasattr(module, 'wk_v'):
#                     module.wk_v = LoRALinear(
#                         module.wk_v,
#                         r=r,
#                         lora_alpha=lora_alpha,
#                         lora_dropout=lora_dropout,
#                         layer_id=f'{current_prefix}/wk_v'
#                     )
#                 if hasattr(module, 'wo'):
#                     module.wo = LoRALinear(
#                         module.wo,
#                         r=r,
#                         lora_alpha=lora_alpha,
#                         lora_dropout=lora_dropout,
#                         layer_id=f'{current_prefix}/wo'
#                     )
#             elif hasattr(module, "name_cells"):
#                 self._replace_modules(
#                     module,
#                     target_modules,
#                     r,
#                     lora_alpha,
#                     lora_dropout,
#                     current_prefix,
#                     block_idx + 1
#                 )

#     def _verify_param_names(self):
#         """验证参数名称唯一性"""
#         param_names = {}
#         duplicates = []
        
#         for param in self.trainable_params():
#             if param.name in param_names:
#                 duplicates.append(f"{param.name} (found in {param_names[param.name]} and current)")
#             else:
#                 param_names[param.name] = param.name
            
#         if duplicates:
#             # 打印所有参数名称以便调试
#             print("\nAll parameter names:")
#             for name in sorted(param_names.keys()):
#                 print(name)
#             raise ValueError(f"Duplicate parameter names found: {duplicates}")
#     def construct(self, *args, **kwargs):
#         return self.base_model(*args, **kwargs)




# class LoRALinear(nn.Cell):
#     """LoRA包装类，用于线性层"""
#     def __init__(self, linear: nn.Dense, r: int = 8, lora_alpha: int = 16, 
#                  lora_dropout: float = 0.05, merge_weights: bool = False,
#                  layer_id: str = None):
#         super().__init__()
#         self.layer_id = layer_id or 'default'
        
#         # 创建linear层并设置唯一名称
#         self.linear = nn.Dense(
#             linear.in_channels,
#             linear.out_channels,
#             has_bias=linear.has_bias,
#             dtype=linear.dtype
#         )
        
#         # 复制原始权重
#         self.linear.weight.set_data(linear.weight)
#         if linear.has_bias and hasattr(linear, 'bias') and linear.bias is not None:
#             self.linear.bias.set_data(linear.bias)
            
#         # 创建LoRA层并设置唯一名称
#         self.lora_A = nn.Dense(
#             linear.in_channels, 
#             r,
#             has_bias=False,
#             dtype=linear.dtype
#         )
#         self.lora_B = nn.Dense(
#             r,
#             linear.out_channels,
#             has_bias=False,
#             dtype=linear.dtype
#         )
        
#         # 设置参数名称
#         if hasattr(self.linear.weight, 'name'):
#             self.linear.weight.name = f'{self.layer_id}.linear.weight'
#         if hasattr(self.linear, 'bias') and self.linear.bias is not None:
#             self.linear.bias.name = f'{self.layer_id}.linear.bias'
#         if hasattr(self.lora_A.weight, 'name'):
#             self.lora_A.weight.name = f'{self.layer_id}.lora_A.weight'
#         if hasattr(self.lora_B.weight, 'name'):
#             self.lora_B.weight.name = f'{self.layer_id}.lora_B.weight'

#         self.r = r
#         self.lora_alpha = lora_alpha
#         self.lora_dropout = nn.Dropout(p=lora_dropout)
#         self.scaling = self.lora_alpha / self.r
#         self.merge_weights = merge_weights
        
#         if merge_weights:
#             self.merge()
#     def construct(self, x):
#         # 原始前向传播
#         base_out = self.linear(x)
        
#         # LoRA前向传播
#         lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        
#         return base_out + lora_out
        
#     def merge(self):
#         """合并LoRA权重到原始权重"""
#         if self.merge_weights:
#             self.linear.weight.set_data(
#                 self.linear.weight + 
#                 self.lora_B.weight @ self.lora_A.weight * self.scaling
#             )
#             self.lora_A.weight.set_data(
#                 ops.zeros_like(self.lora_A.weight)
#             )
#             self.lora_B.weight.set_data(
#                 ops.zeros_like(self.lora_B.weight)
#             )



# class TelechatLoRAModel(nn.Cell):
#     """LoRA包装类，用于整个模型"""
#     def __init__(self,
#                  base_model: TelechatDecodeLayer,
#                  r: int = 8,
#                  lora_alpha: int = 16,
#                  lora_dropout: float = 0.05,
#                  target_modules: list = ["wq", "wk_v", "wo"]):
#         super().__init__()
#         self.base_model = base_model
        
#         # 替换目标模块为LoRA版本
#         self._replace_modules(
#             self.base_model,
#             target_modules,
#             r,
#             lora_alpha,
#             lora_dropout,
#             prefix=""
#         )
        
#         # 验证参数名称唯一性
#         self._verify_param_names()

#     def _replace_modules(self, model, target_modules, r, lora_alpha, lora_dropout, prefix, layer_idx=None):
#         """递归替换目标模块"""
#         for name, module in model.name_cells().items():
#             # 检查是否是transformer层
#             if 'layers.' in name:
#                 # 从名称中提取层号
#                 layer_idx = name.split('.')[1]  # 获取层号
                
#             if isinstance(module, TelechatAttention):
#                 if hasattr(module, 'wq'):
#                     module.wq = LoRALinear(
#                         module.wq,
#                         r=r,
#                         lora_alpha=lora_alpha,
#                         lora_dropout=lora_dropout,
#                         layer_id=f'layer_{layer_idx}_wq' if layer_idx is not None else 'wq'
#                     )
#                 if hasattr(module, 'wk_v'):
#                     module.wk_v = LoRALinear(
#                         module.wk_v,
#                         r=r,
#                         lora_alpha=lora_alpha,
#                         lora_dropout=lora_dropout,
#                         layer_id=f'layer_{layer_idx}_wk_v' if layer_idx is not None else 'wk_v'
#                     )
#                 if hasattr(module, 'wo'):
#                     module.wo = LoRALinear(
#                         module.wo,
#                         r=r,
#                         lora_alpha=lora_alpha,
#                         lora_dropout=lora_dropout,
#                         layer_id=f'layer_{layer_idx}_wo' if layer_idx is not None else 'wo'
#                     )
#             elif hasattr(module, "name_cells"):
#                 self._replace_modules(
#                     module,
#                     target_modules,
#                     r,
#                     lora_alpha,
#                     lora_dropout,
#                     name,
#                     layer_idx
#                 )

#     def _verify_param_names(self):
#         """验证参数名称唯一性"""
#         param_names = {}
#         duplicates = []
        
#         for param in self.trainable_params():
#             if param.name in param_names:
#                 duplicates.append(f"{param.name} (found in {param_names[param.name]} and current)")
#             else:
#                 param_names[param.name] = param.name
            
#         if duplicates:
#             # 打印所有参数名称以便调试
#             print("\nAll parameter names:")
#             for name in sorted(param_names.keys()):
#                 print(name)
#             raise ValueError(f"Duplicate parameter names found: {duplicates}")
#     def construct(self, *args, **kwargs):
#         return self.base_model(*args, **kwargs)


class LoRALinear(nn.Cell):
    """LoRA包装类，用于线性层"""
    def __init__(self, linear: nn.Dense, r: int = 8, lora_alpha: int = 16, 
                 lora_dropout: float = 0.05, merge_weights: bool = False,
                 layer_id: str = None):
        super().__init__()
        self.layer_id = layer_id or 'default'
        
        # 创建linear层并设置唯一名称
        self.linear = nn.Dense(
            linear.in_channels,
            linear.out_channels,
            has_bias=linear.has_bias,
            dtype=linear.dtype
        )
        
        # 复制原始权重
        self.linear.weight.set_data(linear.weight)
        if linear.has_bias and hasattr(linear, 'bias') and linear.bias is not None:
            self.linear.bias.set_data(linear.bias)
            
        # 创建LoRA层并设置唯一名称
        self.lora_A = nn.Dense(
            linear.in_channels, 
            r,
            has_bias=False,
            dtype=linear.dtype
        )
        self.lora_B = nn.Dense(
            r,
            linear.out_channels,
            has_bias=False,
            dtype=linear.dtype
        )
        
        # 设置参数名称，使用完整的路径
        if hasattr(self.linear.weight, 'name'):
            self.linear.weight.name = f'base_model.transformer.{self.layer_id}.linear.weight'
        if hasattr(self.linear, 'bias') and self.linear.bias is not None:
            self.linear.bias.name = f'base_model.transformer.{self.layer_id}.linear.bias'
        if hasattr(self.lora_A.weight, 'name'):
            self.lora_A.weight.name = f'base_model.transformer.{self.layer_id}.lora_A.weight'
        if hasattr(self.lora_B.weight, 'name'):
            self.lora_B.weight.name = f'base_model.transformer.{self.layer_id}.lora_B.weight'

        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.scaling = self.lora_alpha / self.r
        self.merge_weights = merge_weights
        
        if merge_weights:
            self.merge()

    def construct(self, x):
        # 原始前向传播
        base_out = self.linear(x)
        
        # LoRA前向传播
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        
        return base_out + lora_out
        
    def merge(self):
        """合并LoRA权重到原始权重"""
        if self.merge_weights:
            self.linear.weight.set_data(
                self.linear.weight + 
                self.lora_B.weight @ self.lora_A.weight * self.scaling
            )
            self.lora_A.weight.set_data(
                ops.zeros_like(self.lora_A.weight)
            )
            self.lora_B.weight.set_data(
                ops.zeros_like(self.lora_B.weight)
            )


# class TelechatLoRAModel(nn.Cell):
#     """LoRA包装类，用于整个模型"""
#     def __init__(self,
#                  base_model: TelechatDecodeLayer,
#                  r: int = 8,
#                  lora_alpha: int = 16,
#                  lora_dropout: float = 0.05,
#                  target_modules: list = ["wq", "wk_v", "wo"]):
#         super().__init__()
#         self.base_model = base_model
        
#         # 替换目标模块为LoRA版本
#         self._replace_modules(
#             self.base_model,
#             target_modules,
#             r,
#             lora_alpha,
#             lora_dropout,
#             prefix=""
#         )
        
#         # 验证参数名称唯一性
#         self._verify_param_names()

#     def _replace_modules(self, model, target_modules, r, lora_alpha, lora_dropout, prefix, layer_idx=None):
#         """递归替换目标模块"""
#         for name, module in model.name_cells().items():
#             current_path = f"{prefix}.{name}" if prefix else name
            
#             # 通过完整路径匹配层号
#             if 'transformer.layers.' in current_path:
#                 parts = current_path.split('.')
#                 for i, part in enumerate(parts):
#                     if part == 'layers':
#                         layer_idx = parts[i + 1]
#                         break
#             print(f"Current path: {current_path}")
#             print(f"Current module type: {type(module)}")       
                
#             if isinstance(module, TelechatAttention):
#                 if hasattr(module, 'wq'):
#                     module.wq = LoRALinear(
#                         module.wq,
#                         r=r,
#                         lora_alpha=lora_alpha,
#                         lora_dropout=lora_dropout,
#                         layer_id=f'layers.{layer_idx}.attention.wq'
#                     )
#                 if hasattr(module, 'wk_v'):
#                     module.wk_v = LoRALinear(
#                         module.wk_v,
#                         r=r,
#                         lora_alpha=lora_alpha,
#                         lora_dropout=lora_dropout,
#                         layer_id=f'layers.{layer_idx}.attention.wk_v'
#                     )
#                 if hasattr(module, 'wo'):
#                     module.wo = LoRALinear(
#                         module.wo,
#                         r=r,
#                         lora_alpha=lora_alpha,
#                         lora_dropout=lora_dropout,
#                         layer_id=f'layers.{layer_idx}.attention.wo'
#                     )
#             elif hasattr(module, "name_cells"):
#                 self._replace_modules(
#                     module,
#                     target_modules,
#                     r,
#                     lora_alpha,
#                     lora_dropout,
#                     current_path,  # 传递当前完整路径
#                     layer_idx
#                 )

#     def _verify_param_names(self):
#         """验证参数名称唯一性"""
#         param_names = {}
#         duplicates = []
        
#         for param in self.trainable_params():
#             if param.name in param_names:
#                 duplicates.append(f"{param.name} (found in {param_names[param.name]} and current)")
#             else:
#                 param_names[param.name] = param.name
            
#         if duplicates:
#             # 打印所有参数名称以便调试
#             print("\nAll parameter names:")
#             for name in sorted(param_names.keys()):
#                 print(name)
#             raise ValueError(f"Duplicate parameter names found: {duplicates}")

#     def construct(self, *args, **kwargs):
#         return self.base_model(*args, **kwargs)


class TelechatLoRAModel(nn.Cell):
    """LoRA包装类，用于整个模型"""
    def __init__(self,
                 base_model: TelechatDecodeLayer,
                 r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 target_modules: list = ["wq", "wk_v", "wo"]):
        super().__init__()
        self.base_model = base_model
        
        # 替换目标模块为LoRA版本
        self._replace_modules(
            self.base_model,
            target_modules,
            r,
            lora_alpha,
            lora_dropout,
            prefix="base_model"  # 修改初始前缀为base_model
        )
        
        # 验证参数名称唯一性
        self._verify_param_names()

    def _replace_modules(self, model, target_modules, r, lora_alpha, lora_dropout, prefix, layer_idx=None):
        """递归替换目标模块"""
        for name, module in model.name_cells().items():
            current_path = f"{prefix}.{name}" if prefix else name
            
            # 调试信息
            print(f"Current path: {current_path}")
            print(f"Current module type: {type(module)}")
            
            # 更新layer_idx
            new_layer_idx = None
            if 'transformer.layers.' in current_path:
                parts = current_path.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        new_layer_idx = parts[i + 1]
                        break
                layer_idx = new_layer_idx
                print(f"Found layer_idx: {layer_idx} at path: {current_path}")
                    
            if isinstance(module, TelechatAttention):
                if layer_idx is None:
                    print(f"Warning: Found TelechatAttention but no layer_idx at path {current_path}")
                    continue
                    
                print(f"Replacing attention module in layer {layer_idx}")
                
                # 保存原始参数
                original_wq = module.wq if hasattr(module, 'wq') else None
                original_wk_v = module.wk_v if hasattr(module, 'wk_v') else None
                original_wo = module.wo if hasattr(module, 'wo') else None
                
                # 删除原始属性
                if hasattr(module, 'wq'): delattr(module, 'wq')
                if hasattr(module, 'wk_v'): delattr(module, 'wk_v')
                if hasattr(module, 'wo'): delattr(module, 'wo')
                
                # 创建新的LoRA层
                if original_wq is not None:
                    module.wq = LoRALinear(
                        original_wq,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        layer_id=f'base_model.transformer.layers.{layer_idx}.attention.wq'
                    )
                
                if original_wk_v is not None:
                    module.wk_v = LoRALinear(
                        original_wk_v,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        layer_id=f'base_model.transformer.layers.{layer_idx}.attention.wk_v'
                    )
                    
                if original_wo is not None:
                    module.wo = LoRALinear(
                        original_wo,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        layer_id=f'base_model.transformer.layers.{layer_idx}.attention.wo'
                    )
            
            # 如果模块还有子模块，继续递归
            if hasattr(module, "name_cells"):
                self._replace_modules(
                    module,
                    target_modules,
                    r,
                    lora_alpha,
                    lora_dropout,
                    current_path,  # 传递当前完整路径
                    layer_idx
                )

    def _verify_param_names(self):
        """验证参数名称唯一性"""
        param_names = {}
        duplicates = []
        
        for param in self.trainable_params():
            if param.name in param_names:
                duplicates.append(f"{param.name} (found in {param_names[param.name]} and current)")
            else:
                param_names[param.name] = param.name
            
        if duplicates:
            # 打印所有参数名称以便调试
            print("\nAll parameter names:")
            for name in sorted(param_names.keys()):
                print(name)
            raise ValueError(f"Duplicate parameter names found: {duplicates}")

    def construct(self, *args, **kwargs):
        """前向传播"""
        return self.base_model(*args, **kwargs)