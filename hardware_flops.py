


def main():
    hidden_sizes = [6144, 12288, 20480, 25600]
    s = 2048

    estimated_ratios = [1 + s / (6 * h) for h in hidden_sizes]
    improved_estimation = [1 + s / (18 * h) for h in hidden_sizes]

    model_utils = [41.5, 51.4, 56.0, 56.3]
    hardware_utils = [43.7, 52.8, 57.0, 57.0]

    observed_ratios = [h/m for m,h in zip(model_utils, hardware_utils)]

    print("Observed estimation", observed_ratios)
    print("Paper estimation", estimated_ratios)
    print("Improved estimation", improved_estimation)

if __name__ == "__main__":
    main()