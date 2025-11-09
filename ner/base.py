from abc import ABC, abstractmethod
from typing import List, Dict


class NERModel(ABC):
    """Abstract base class defining a consistent interface for NER models."""

    @abstractmethod
    def predict(self, text: str) -> List[Dict]:
        """
        Runs inference and returns a normalized output:
        [
          {"entity": "Apple", "label": "ORG", "score": 0.99, "start": 0, "end": 5},
          ...
        ]
        """
        pass
