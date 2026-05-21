import yaml
from pathlib import Path
from path_utils import PathUtils


def load_config(config_name: str) -> dict:
    path = PathUtils.to_absolute(f"config/{config_name}.yml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


CONFIG_MODULES = ["rag", "chroma", "prompts", "agent"]
configs = {name: load_config(name) for name in CONFIG_MODULES}

rag_conf = configs["rag"]
agent_conf = configs["agent"]
print(rag_conf)
