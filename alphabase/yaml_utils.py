import yaml

def load_yaml(filename)->dict:
    with open(filename) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    return settings

def save_yaml(filename, settings):
    with open(filename, "w") as file:
        yaml.dump(settings, file)
