alias GGUF_MAGIC             = 0x46554747  # "GGUF"
alias GGUF_VERSION           = 3
alias GGUF_DEFAULT_ALIGNMENT = 32

@register_passable("trivial")
struct Keys:
    alias GENERAL_ARCHITECTURE = "general.architecture"
    alias GENERAL_QUANTIZATION_VERSION = "general.quantization_version"
    alias GENERAL_ALIGNMENT = "general.alignment"
    alias GENERAL_NAME = "general.name"
    alias GENERAL_AUTHOR = "general.author"
    alias GENERAL_URL = "general.url"
    alias GENERAL_DESCRIPTION = "general.description"
    alias GENERAL_LICENSE = "general.license"
    alias GENERAL_SOURCE_URL = "general.source.url"
    alias GENERAL_SOURCE_HF_REPO = "general.source.huggingface.repository"
    alias GENERAL_FILE_TYPE = "general.file_type"

    alias LLM_CONTEXT_LENGTH = "{arch}.context_length"
    alias LLM_EMBEDDING_LENGTH = "{arch}.embedding_length"
    alias LLM_BLOCK_COUNT = "{arch}.block_count"
    alias LLM_FEED_FORWARD_LENGTH = "{arch}.feed_forward_length"
    alias LLM_USE_PARALLEL_RESIDUAL = "{arch}.use_parallel_residual"
    alias LLM_TENSOR_DATA_LAYOUT = "{arch}.tensor_data_layout"
    alias LLM_EXPERT_COUNT = "{arch}.expert_count"
    alias LLM_EXPERT_USED_COUNT = "{arch}.expert_used_count"

    alias ATTENTION_HEAD_COUNT = "{arch}.attention.head_count"
    alias ATTENTION_HEAD_COUNT_KV = "{arch}.attention.head_count_kv"
    alias ATTENTION_MAX_ALIBI_BIAS = "{arch}.attention.max_alibi_bias"
    alias ATTENTION_CLAMP_KQV = "{arch}.attention.clamp_kqv"
    alias ATTENTION_LAYERNORM_EPS = "{arch}.attention.layer_norm_epsilon"
    alias ATTENTION_LAYERNORM_RMS_EPS = "{arch}.attention.layer_norm_rms_epsilon"

    alias ROPE_DIMENSION_COUNT = "{arch}.rope.dimension_count"
    alias ROPE_FREQ_BASE = "{arch}.rope.freq_base"
    alias ROPE_SCALING_TYPE = "{arch}.rope.scaling.type"
    alias ROPE_SCALING_FACTOR = "{arch}.rope.scaling.factor"
    alias ROPE_SCALING_ORIG_CTX_LEN = "{arch}.rope.scaling.original_context_length"
    alias ROPE_SCALING_FINETUNED = "{arch}.rope.scaling.finetuned"

    alias TOKENIZER_MODEL = "tokenizer.ggml.model"
    alias TOKENIZER_LIST = "tokenizer.ggml.tokens"
    alias TOKENIZER_TOKEN_TYPE = "tokenizer.ggml.token_type"
    alias TOKENIZER_SCORES = "tokenizer.ggml.scores"
    alias TOKENIZER_MERGES = "tokenizer.ggml.merges"
    alias TOKENIZER_BOS_ID = "tokenizer.ggml.bos_token_id"
    alias TOKENIZER_EOS_ID = "tokenizer.ggml.eos_token_id"
    alias TOKENIZER_UNK_ID = "tokenizer.ggml.unknown_token_id"
    alias TOKENIZER_SEP_ID = "tokenizer.ggml.seperator_token_id"
    alias TOKENIZER_PAD_ID = "tokenizer.ggml.padding_token_id"
    alias TOKENIZER_ADD_BOS = "tokenizer.ggml.add_bos_token"
    alias TOKENIZER_ADD_EOS = "tokenizer.ggml.add_eos_token"
    alias TOKENIZER_HF_JSON = "tokenizer.huggingface.json"
    alias TOKENIZER_RWKV = "tokenizer.rwkv.world"
    alias TOKENIZER_CHAT_TEMPLATE = "tokenizer.chat_template"

#
# recommended mapping of model tensor names for storage in gguf
#

@register_passable("trivial")
struct MODEL_ARCH:
    alias LLAMA     = "llama"
    alias FALCON    = "falcon"
    alias BAICHUAN  = "baichuan"
    alias GPT2      = "gpt2"
    alias GPTJ      = "gptj"
    alias GPTNEOX   = "gptneox"
    alias MPT       = "mpt"
    alias STARCODER = "starcoder"
    alias PERSIMMON = "persimmon"
    alias REFACT    = "refact"
    alias BERT      = "bert"
    alias BLOOM     = "bloom"
    alias STABLELM  = "stablelm"
    alias QWEN      = "qwen"
    alias PHI2      = "phi2"


@register_passable("trivial")
struct MODEL_TENSOR:
    alias TOKEN_EMBD      = "token_embd"
    alias TOKEN_EMBD_NORM = "token_embd_norm"
    alias TOKEN_TYPES     = "token_types"
    alias POS_EMBD        = "pos_embd"
    alias OUTPUT          = "output"
    alias OUTPUT_NORM     = "output_norm"
    alias ROPE_FREQS      = "rope_freqs"
    alias ATTN_Q          = "attn_q"
    alias ATTN_K          = "attn_k"
    alias ATTN_V          = "attn_v"
    alias ATTN_QKV        = "attn_qkv"
    alias ATTN_OUT        = "attn_out"
    alias ATTN_NORM       = "attn_norm"
    alias ATTN_NORM_2     = "attn_norm_2"
    alias ATTN_ROT_EMBD   = "attn_rot_embd"
    alias FFN_GATE_INP    = "ffn_gate_inp"
    alias FFN_NORM        = "ffn_norm"
    alias FFN_GATE        = "ffn_gate"
    alias FFN_DOWN        = "ffn_down"
    alias FFN_UP          = "ffn_up"
    alias FFN_GATE_EXP    = "ffn_gate_exp"
    alias FFN_DOWN_EXP    = "ffn_down_exp"
    alias FFN_UP_EXP      = "ffn_up_exp"
    alias ATTN_Q_NORM     = "attn_q_norm"
    alias ATTN_K_NORM     = "attn_k_norm"



@register_passable("trivial")
struct MODEL_ARCH_NAMES:
    alias LLAMA = "llama"
    alias FALCON = "falcon"
    alias BAICHUAN = "baichuan"
    alias GPT2 = "gpt2"
    alias GPTJ = "gptj"
    alias GPTNEOX = "gptneox"
    alias MPT = "mpt"
    alias STARCODER = "starcoder"
    alias PERSIMMON = "persimmon"
    alias REFACT = "refact"
    alias BERT = "bert"
    alias BLOOM = "bloom"
    alias STABLELM = "stablelm"
    alias QWEN = "qwen"
    alias PHI2 = "phi2"


@register_passable("trivial")
struct TENSOR_NAMES:
    alias TOKEN_EMBD = "token_embd"
    alias TOKEN_EMBD_NORM = "token_embd_norm"
    alias TOKEN_TYPES = "token_types"
    alias POS_EMBD = "position_embd"
    alias OUTPUT_NORM = "output_norm"
    alias OUTPUT = "output"
    alias ROPE_FREQS = "rope_freqs"
    alias ATTN_NORM = "blk.{bid}.attn_norm"
    alias ATTN_NORM_2 = "blk.{bid}.attn_norm_2"
    alias ATTN_QKV = "blk.{bid}.attn_qkv"
    alias ATTN_Q = "blk.{bid}.attn_q"
    alias ATTN_K = "blk.{bid}.attn_k"
    alias ATTN_V = "blk.{bid}.attn_v"
    alias ATTN_OUT = "blk.{bid}.attn_output"
    alias ATTN_ROT_EMBD = "blk.{bid}.attn_rot_embd"
    alias ATTN_Q_NORM = "blk.{bid}.attn_q_norm"
    alias ATTN_K_NORM = "blk.{bid}.attn_k_norm"
    alias FFN_GATE_INP = "blk.{bid}.ffn_gate_inp"
    alias FFN_NORM = "blk.{bid}.ffn_norm"
    alias FFN_GATE = "blk.{bid}.ffn_gate"
    alias FFN_DOWN = "blk.{bid}.ffn_down"
    alias FFN_UP = "blk.{bid}.ffn_up"
    alias FFN_GATE_EXP = "blk.{bid}.ffn_gate.{xid}"
    alias FFN_DOWN_EXP = "blk.{bid}.ffn_down.{xid}"
    alias FFN_UP_EXP = "blk.{bid}.ffn_up.{xid}"

from python import Python


@register_passable("trivial")
struct GGUFValueType:
    alias UINT8 = 0
    alias INT8 = 1
    alias UINT16 = 2
    alias INT16 = 3
    alias UINT32 = 4
    alias INT32 = 5
    alias FLOAT32 = 6
    alias BOOL = 7
    alias STRING = 8
    alias ARRAY = 9
    alias UINT64 = 10
    alias INT64 = 11
    alias FLOAT64 = 12

    @staticmethod
    def get_type(val: PythonObject) -> Int:
        let builtins = Python.import_module("builtins")
        let isinstance = builtins.isinstance
        if isinstance(val, builtins.str) or isinstance(val, builtins.bytes) or isinstance(val, builtins.bytearray):
            return GGUFValueType.STRING
        elif isinstance(val, builtins.list):
            return GGUFValueType.ARRAY
        elif isinstance(val, builtins.float):
            return GGUFValueType.FLOAT32
        elif isinstance(val, builtins.bool):
            return GGUFValueType.BOOL
        elif isinstance(val, builtins.int):
            return GGUFValueType.INT32
        # TODO: need help with 64-bit types in Python
        else:
            raise "Unsupported type"
    
    @staticmethod
    fn get_dtype(val : Int) raises -> DType:
        if val == 0:
            return DType.uint8
        elif val == 1:
            return DType.int8
        elif val == 2:
            return DType.uint16
        elif val == 3:
            return DType.int16
        elif val == 4:
            return DType.uint32
        elif val == 5:
            return DType.int32
        elif val == 6:
            return DType.float32
        elif val == 7:
            return DType.bool
        elif val == 10:
            return DType.uint64
        elif val == 11:
            return DType.int64
        elif val == 12:
            return DType.float64
        else:
            raise "Unsupported type"

    

@register_passable("trivial")
struct GGMLQuantizationType:
    alias F32 = 0
    alias F16 = 1
    alias Q4_0 = 2
    alias Q4_1 = 3
    alias Q5_0 = 6
    alias Q5_1 = 7
    alias Q8_0 = 8
    alias Q8_1 = 9
    alias Q2_K = 10
    alias Q3_K = 11
    alias Q4_K = 12
    alias Q5_K = 13
    alias Q6_K = 14
    alias Q8_K = 15


# Note: Does not support GGML_QKK_64
alias QK_K = 256


# Items here are (block size, type size)
@register_passable("trivial")
struct GGML_QUANT_SIZES:
    alias F32 = (1, 4)
    alias F16 = (1, 2)
    alias Q4_0 = (32, 2 + 16)
    alias Q4_1 = (32, 2 + 2 + 16)
    alias Q5_0 = (32, 2 + 4 + 16)
    alias Q5_1 = (32, 2 + 2 + 4 + 16)
    alias Q8_0 = (32, 2 + 32)
    alias Q8_1 = (32, 4 + 4 + 32)
    alias Q2_K = (256, 2 + 2 + QK_K // 16 + QK_K // 4)
    alias Q3_K = (256, 2 + QK_K // 4 + QK_K // 8 + 12)
    alias Q4_K = (256, 2 + 2 + QK_K // 2 + 12)
    alias Q5_K = (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12)
    alias Q6_K = (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16)
    alias Q8_K = (256, 4 + QK_K + QK_K // 8)
