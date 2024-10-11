import _pickle as pickle
import yaml


def load_config(config_filename):
    with open(config_filename, 'r') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        return cfg


def save_config(config_filename, config):
    with open(config_filename, 'w') as f:
        f_str = yaml.dump(config, default_flow_style=False)
        f.write(f_str)


def load_object(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_object(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_objects(path_list):
    return [load_object(p) for p in path_list]


def save_objects(obj_list, path_list):
    for obj, path in zip(obj_list, path_list):
        save_object(obj, path)
