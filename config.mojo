struct MixtralConfig:
    var vocab_size: Int
    var hidden_size: Int
    var intermediate_size: Int
    var num_hidden_layers: Int
    var num_attention_heads: Int
    var num_key_value_heads: Int
    var hidden_act: String
    var max_position_embeddings: Int
    var initializer_range: FloatLiteral
    var rms_norm_eps: FloatLiteral
    var use_cache: Bool
    var pad_token_id: Int
    var bos_token_id: Int
    var eos_token_id: Int
    var tie_word_embeddings: Bool
    var rope_theta: FloatLiteral
    var sliding_window: Int
    var attention_dropout: FloatLiteral
    var num_experts_per_tok: Int
    var num_local_experts: Int
    var output_router_logits: Bool
    var router_aux_loss_coef: FloatLiteral

    def __init__(
        inout self,
        vocab_size: Int = 32000,
        hidden_size: Int = 4096,
        intermediate_size: Int = 14336,
        num_hidden_layers: Int = 32,
        num_attention_heads: Int = 32,
        num_key_value_heads: Int = 8,
        hidden_act: String = "silu",
        max_position_embeddings: Int = 4096 * 32,
        initializer_range: FloatLiteral = 0.02,
        rms_norm_eps: FloatLiteral = 1e-5,
        use_cache: Bool = True,
        pad_token_id: Int = -1,
        bos_token_id: Int = 1,
        eos_token_id: Int = 2,
        tie_word_embeddings: Bool = False,
        rope_theta: FloatLiteral = 1e6,
        sliding_window: Int = 4096,
        attention_dropout: FloatLiteral = 0.0,
        num_experts_per_tok: Int = 2,
        num_local_experts: Int = 8,
        output_router_logits: Bool = False,
        router_aux_loss_coef: FloatLiteral = 0.001,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
