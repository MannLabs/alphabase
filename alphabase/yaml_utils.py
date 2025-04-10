import yaml


def load_yaml(filename) -> dict:
    with open(filename) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def save_yaml(filename, settings):
    with open(filename, "w") as file:
        yaml.dump(settings, file, sort_keys=False)
