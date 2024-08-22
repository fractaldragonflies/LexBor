"""
Settings for different models.
"""

import attr


@attr.s  # pylint: disable=too-many-instance-attributes,too-few-public-methods
class BaseSettings:
    """
    Base configuration settings.
    """

    # Corpus and document qualifiers.
    language = attr.ib(default="")
    series = attr.ib(default="")
    basis = attr.ib(default="all")  # native, loan, all, both
    detect_type = attr.ib(default="dual")  # native, dual, direct, auto,
    form = attr.ib(default="Tokens")
    dlm_is_plus = attr.ib(default=False)

    test_split = attr.ib(default=0.15)
    val_split = attr.ib(default=0.15)
    min_count = attr.ib(default=50)

    verbose = attr.ib(default=1)

    print_summary = attr.ib(default=False, metadata={"deprecated": True})
    plot_model = attr.ib(default=False, metadata={"deprecated": True})
    plot_dpi = attr.ib(default=400, metadata={"deprecated": True})

    output_path = attr.ib(default="./output")


@attr.s  # pylint: disable=too-many-instance-attributes,too-few-public-methods
class NeuralSettings(BaseSettings):
    """
    Neural model settings.
    """
    tf_verbose = attr.ib(default=0)

    # Data handling.
    batch_size = attr.ib(default=32)  # 32 previously # larger is better for scarce borrowing.
    skip_step = attr.ib(default=3)  # 0 for direct model, small value 3, 5, 7 for entropy model.
    token_maxlen = attr.ib(default=30)
    output_encoding = attr.ib(default='sparse')  # sparse, 1hot

    # Detector controls.
    model_type = attr.ib(default="multiheadattention")
    # recurrent, recurrentattention, multiheadattention, recurrent_lm_for, recurrent_lm_for_back
    fraction = attr.ib(default=0.80)  # For Native model. Use complement with util.
    prediction_policy = attr.ib(default="zero")  # zero, accuracy, fscore
    fscore_beta = attr.ib(default=1.0)
    use_class_weight = attr.ib(default=True)  # direct model. Weights classes inversely.
    use_equal_steps = attr.ib(default=False)  # entropy model. Adjust epochs for equal classes.
    # use_class_bias = attr.ib(default=False)


@attr.s  # pylint: disable=too-many-instance-attributes,too-few-public-methods
class RecurrentSettings(NeuralSettings):
    """
    Recurrent model settings.
    """

    # Architecture parameters
    embedding_len = attr.ib(default=32)  # previously 32
    rnn_output_len = attr.ib(default=32)  # previously 32
    rnn_cell_type = attr.ib(default="LSTM")  # GRU, LSTM
    rnn_levels = attr.ib(default=1)  # 1, 2
    merge_embedding = attr.ib(default=True)
    # Attention architecture parameters
    attention_type = attr.ib(default="MUL")  # ADD, MUL
    attention_causal = attr.ib(default=True)  # True for entropy.
    transformer_stuff = attr.ib(default=True)
    num_heads = attr.ib(default=4)
    positional_encoding = attr.ib(default="TRIG")  # EMB, TRIG
    flatten_attention = attr.ib(default=False)
    attention_levels = attr.ib(default=1)  # 1, 2

    # Dropout and regulation parameters
    embedding_dropout = attr.ib(default=0.0)
    decoding_dropout = attr.ib(default=0.0)
    logit_dropout = attr.ib(default=0.0)
    kernel_l2 = attr.ib(default=0.0)
    recurrent_l2 = attr.ib(default=0.0)
    rnn_activity_l2 = attr.ib(default=0.0)
    recurrent_dropout = attr.ib(default=0.0)
    rnn_output_dropout = attr.ib(default=0.0)
    attention_dropout = attr.ib(default=0.0)
    transformer_dropout = attr.ib(default=0.0)
    merge_embedding_dropout = attr.ib(default=0.0)

    # Model fitting parameters
    epochs = attr.ib(default=45)
    learning_rate = attr.ib(default=0.0055)
    learning_rate_decay = attr.ib(default=0.95)
    learning_rate_schedule = attr.ib(default='TR')
    # TR = transformer, EX = exponential, IT = inverse time, CU = cubic, None = fixed, SP = special.
    learning_rate_warmup = attr.ib(default=90)
    restore_best_weights = attr.ib(default=False)  # False for experiments.
    early_stopping = attr.ib(default=False)  # False for experiments.
    early_stopping_F1 = attr.ib(default=False)  # Use F1score for early stopping.
    patience = attr.ib(default=3)
    len_power = attr.ib(default=1)  # For native since compare entropy to fixed limit.


@attr.s  # pylint: disable=too-many-instance-attributes,too-few-public-methods
class MarkovSettings(BaseSettings):
    """
    Markov model settings.
    """
    model = attr.ib(default="kni")
    order = attr.ib(default=3)
    p = attr.ib(default=0.8)  # pylint: disable=invalid-name
    z = attr.ib(default=4.5)  # pylint: disable=invalid-name
    smoothing = attr.ib(default=0.2)
    direction = attr.ib(default="forward")
