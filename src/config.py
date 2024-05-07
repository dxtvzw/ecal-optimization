import yaml
from types import SimpleNamespace


def parse(d):
    x = SimpleNamespace()
    _ = [setattr(x, k,
                 parse(v) if isinstance(v, dict)
                 else [parse(e) for e in v] if isinstance(v, list)
                 else v) for k, v in d.items()]
    return x


def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

    cfg = parse(cfg_dict)
    return cfg, cfg_dict