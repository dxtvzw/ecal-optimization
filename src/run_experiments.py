import yaml
import subprocess
import os
import signal
import copy
import re


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


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    if config['experiment_id'] == 1:
        experiment_1(config)


if __name__ == "__main__":
    main()
