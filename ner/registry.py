import yaml
from ner.gliner_model import GLiNERModel
from ner.hf_model import HFNERModel


class ModelRegistry:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.config = cfg["models"]
        self.models = {}

    def get_model(self, name: str):
        if name not in self.config:
            raise KeyError(f"Model '{name}' not found in config.")
        if name not in self.models:
            self.models[name] = self._load_model(name)
        return self.models[name]

    def _load_model(self, name: str):
        conf = self.config[name]
        if conf["type"] == "gliner":
            return GLiNERModel(conf["path"], conf["labels"])
        elif conf["type"] == "hf":
            return HFNERModel(conf["model_name"])
        else:
            raise ValueError(f"Unsupported model type: {conf['type']}")

    def list_models(self):
        return [
            {
                "name": name,
                "type": conf["type"],
                "version": conf.get("version"),
                "labels": conf.get("labels", None),
            }
            for name, conf in self.config.items()
        ]
