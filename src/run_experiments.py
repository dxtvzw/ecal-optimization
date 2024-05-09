import yaml
import subprocess
import os
import signal
import copy
import re
import argparse


def extract_dimensions(filename):
    match = re.search(r'\d+x\d+', filename)
    if match:
        dimensions = match.group(0)
        x, y = dimensions.split('x')
        return int(x), int(y)
    else:
        return None


proc_list = []


def run_experiment(config_path, group_name):
    os.environ["WANDB_RUN_GROUP"] = group_name
    proc = subprocess.Popen(["torchrun", "--nproc_per_node", "8", "main.py", "--config", config_path])
    proc_list.append(proc)
    proc.wait()


def signal_handler(sig, frame):
    # Handle termination signal (SIGINT or SIGTERM)
    for proc in proc_list:
        proc.terminate()
    exit(0)


def experiment_1(config):
    model_tags = [
        "SumModel",
        "LinearModel",
        "MyResnet18",
        "MyCNN",
        "MyViT",
    ]

    data_files = [
        "64k_wpc_10x10_v2.npz",
        "64k_wpc_15x15.npz",
        "64k_wpc_20x20.npz",
        "64k_wpc_25x25.npz",
        "64k_wpc_30x30.npz",
        "64k_wpc_40x40.npz",
    ]

    for model_tag in model_tags:
        for data_file in data_files:
            cur_config = copy.deepcopy(config)
            cur_config['model']['tag'] = model_tag
            cur_config['paths']['data_file'] = data_file

            temp_config_path = 'temp_config.yaml'
            with open(temp_config_path, 'w') as temp_config_file:
                yaml.safe_dump(cur_config, temp_config_file)

            height, width = extract_dimensions(data_file)

            run_experiment(temp_config_path, f"exp1_{model_tag}_{height}x{width}")

            os.remove(temp_config_path)


def experiment_2(config):
    model_tags = [
        "MyViT",
    ]

    data_files = [
        "64k_wpc_10x10_v2.npz",
        "64k_wpc_15x15.npz",
        "64k_wpc_20x20.npz",
        "64k_wpc_25x25.npz",
        "64k_wpc_30x30.npz",
        "64k_wpc_40x40.npz",
    ]

    eng_losses = [
        "RMSE_E",
        "MAE_E",
        "RMSLE",
        "RMSE",
    ]

    for model_tag in model_tags:
        for data_file in data_files:
            for loss_fn_eng in eng_losses:
                cur_config = copy.deepcopy(config)
                cur_config['model']['tag'] = model_tag
                cur_config['paths']['data_file'] = data_file
                cur_config['training']['loss_fn_eng'] = loss_fn_eng

                temp_config_path = 'temp_config.yaml'
                with open(temp_config_path, 'w') as temp_config_file:
                    yaml.safe_dump(cur_config, temp_config_file)

                height, width = extract_dimensions(data_file)

                run_experiment(temp_config_path, f"exp2_{height}x{width}_{loss_fn_eng}")

                os.remove(temp_config_path)


def experiment_3(config):
    model_tags = [
        "MyViT",
    ]

    data_files = [
        "64k_wpc_10x10_v2.npz",
        "64k_wpc_25x25.npz",
        "64k_wpc_40x40.npz",
    ]

    eng_weights = []
    for i in range(0, 11):
        eng_weights.append(i / 10)

    for model_tag in model_tags:
        for data_file in data_files:
            for eng_weight in eng_weights:
                cur_config = copy.deepcopy(config)
                cur_config['model']['tag'] = model_tag
                cur_config['paths']['data_file'] = data_file
                cur_config['training']['eng_loss_weight'] = eng_weight

                temp_config_path = 'temp_config.yaml'
                with open(temp_config_path, 'w') as temp_config_file:
                    yaml.safe_dump(cur_config, temp_config_file)

                height, width = extract_dimensions(data_file)

                run_experiment(temp_config_path, f"exp3_{height}x{width}_{eng_weight}")

                os.remove(temp_config_path)


def experiment_4(config):
    model_tags = [
        "MyViT",
    ]

    data_files = [
        "64k_wpc_25x25.npz",
    ]

    num_layers = [2, 4, 6]

    hidden_dims = [8, 16, 32, 64]

    num_heads = [2, 4, 8]

    for model_tag in model_tags:
        for data_file in data_files:
            for num_layer in num_layers:
                for hidden_dim in hidden_dims:
                    for n_heads in num_heads:
                        cur_config = copy.deepcopy(config)
                        cur_config['model']['tag'] = model_tag
                        cur_config['paths']['data_file'] = data_file
                        cur_config['model']['num_layers'] = num_layer
                        cur_config['model']['hidden_dim'] = hidden_dim
                        cur_config['model']['num_heads'] = n_heads

                        cur_config['model']['mlp_dim'] = hidden_dim * 4

                        temp_config_path = 'temp_config.yaml'
                        with open(temp_config_path, 'w') as temp_config_file:
                            yaml.safe_dump(cur_config, temp_config_file)

                        run_experiment(temp_config_path, f"exp4_{num_layer}_{hidden_dim}_{n_heads}")

                        os.remove(temp_config_path)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Experiment runner")
        parser.add_argument("--exp_id", type=int, required=True, help="Experiment id to run")
        return parser.parse_args()

    args = parse_arguments()

    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    if args.exp_id == 1:
        experiment_1(config)
    elif args.exp_id == 2:
        experiment_2(config)
    elif args.exp_id == 3:
        experiment_3(config)
    elif args.exp_id == 4:
        experiment_4(config)
    else:
        raise ValueError(f"Invalid experiment id: {args.exp_id}")


if __name__ == "__main__":
    main()
