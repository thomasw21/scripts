from transformers import AutoConfig

def main():
    model_name = "google/t5-v1_1-xl"
    config = AutoConfig.from_pretrained(model_name)

    # Iteration parameters:
    batch_size = 2048
    input_length = 512
    target_length = 114
    checkpoint_activation = True

    # Experiment result
    iter_per_sec = 0.23
    gpus = 64

    """
    Some comments about the current computation script:
     - we ignore all layer norms
     - we ignore changes in precision
    """

    FLOP_per_iteration = 0

    ### Encoder FLOPS
    # self attention QKV matrix multiplication
    FLOP_per_iteration += config.num_layers * batch_size * input_length * 2 * 3  * config.d_model ** 2
    # self attention attention matrix computation (we ignore softmax computation)
    FLOP_per_iteration += config.num_layers * batch_size * config.d_model * 2 * input_length ** 2
    # self attention apply attention
    FLOP_per_iteration += config.num_layers * batch_size * config.d_model * 2 * input_length ** 2
    # self attention output
    FLOP_per_iteration += config.num_layers * batch_size * input_length * 2 * config.d_model ** 2
    # MLP first layer (t5 uses GLU, so first layer is twice the d_ff)
    FLOP_per_iteration += config.num_layers * batch_size * input_length * 2 * config.d_model * 2 * config.d_ff
    # MLP second layer
    FLOP_per_iteration += config.num_layers * batch_size * input_length * 2 * config.d_model * config.d_ff

    ### Decoder FLOPS
    # self attention QKV matrix multiplication
    FLOP_per_iteration += config.num_layers * batch_size * target_length * 2 * 3  * config.d_model ** 2
    # self attention QKV attention matrix computation (we ignore softmax computation)
    FLOP_per_iteration += config.num_layers * batch_size * config.d_model * 2 * target_length ** 2
    # self attention apply attention
    FLOP_per_iteration += config.num_layers * batch_size * config.d_model * 2 * target_length ** 2
    # self attention output
    FLOP_per_iteration += config.num_layers * batch_size * target_length * 2 * config.d_model ** 2
    # cross attention Q matrix multiplication
    FLOP_per_iteration += config.num_layers * batch_size * target_length * 2 * config.d_model ** 2
    # cross attention KV matrix multiplication
    FLOP_per_iteration += config.num_layers * batch_size * input_length * 2 * 2 * config.d_model ** 2
    # cross attention QKV attention matrix computation (we ignore softmax computation)
    FLOP_per_iteration += config.num_layers * batch_size * config.d_model * 2 * target_length * input_length
    # cross attention apply attention
    FLOP_per_iteration += config.num_layers * batch_size * config.d_model * 2 * target_length * input_length
    # cross attention output
    FLOP_per_iteration += config.num_layers * batch_size * target_length * 2 * config.d_model ** 2
    # MLP first layer (t5 uses GLU, so first layer is twice the d_ff)
    FLOP_per_iteration += config.num_layers * batch_size * target_length * 2 * config.d_model * 2 * config.d_ff
    # MLP second layer
    FLOP_per_iteration += config.num_layers * batch_size * target_length * 2 * config.d_model * config.d_ff
    # Logits
    FLOP_per_iteration += batch_size * target_length * 2 * config.d_model * config.vocab_size

    # Backward pass (essentially twice the TFLOPs)
    # 1 for forward pass and 2 for backward pass (+ 1 for activation checkpoint)
    if checkpoint_activation:
        FLOP_per_iteration += (2 + 1) * FLOP_per_iteration
    else:
        FLOP_per_iteration += 2 * FLOP_per_iteration

    print(f"Total TFLOP per iteration: {FLOP_per_iteration / (10 ** 12):.2f}")
    if iter_per_sec is not None:
        TFLOPs = FLOP_per_iteration * iter_per_sec / (10 ** 12)
        print(f"TFLOPs (speed): {TFLOPs:.2f}")
        if gpus is not None:
            print(f"TFLOPs per gpu: {TFLOPs / gpus:.2f}")

if __name__ == "__main__":
    main()