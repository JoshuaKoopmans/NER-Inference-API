import os
from gliner import GLiNER
from ner.base import NERModel


class GLiNERModel(NERModel):
    def __init__(self, path: str, labels: list[str]):
        self.path = self._resolve_path(path)
        self.labels = labels
        self.model = None

    def _resolve_path(self, path: str) -> str:
        # if path looks like a local directory, expand it
        if os.path.exists(os.path.expanduser(path)):
            return os.path.expanduser(path)
        # else treat as HF repo ID
        return path

    def _ensure_loaded(self):
        if self.model is None:
            self.model = GLiNER.from_pretrained(self.path)

    def predict(self, text: str):
        self._ensure_loaded()
        results = self.model.predict_entities(text, self.labels)
        return [
            {
                "entity": r["text"],
                "label": r["label"],
                "score": float(r["score"]),
                "start": int(r["start"]),
                "end": int(r["end"]),
            }
            for r in results
        ]
