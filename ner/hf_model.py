from transformers import pipeline
from ner.base import NERModel


class HFNERModel(NERModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pipe = None  # Lazy load

    def _ensure_loaded(self):
        if self.pipe is None:
            self.pipe = pipeline(
                "ner", model=self.model_name, aggregation_strategy="simple"
            )

    def predict(self, text: str):
        self._ensure_loaded()
        results = self.pipe(text)
        return [
            {
                "entity": r["word"],
                "label": r["entity_group"],
                "score": float(r["score"]),
                "start": int(r["start"]),
                "end": int(r["end"]),
            }
            for r in results
        ]
