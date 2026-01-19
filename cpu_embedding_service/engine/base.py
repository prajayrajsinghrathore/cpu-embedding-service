from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingEngine(ABC):
    @abstractmethod
    def load_model(self, model_name: str) -> None:
        pass

    @abstractmethod
    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        truncate: bool = True
    ) -> List[List[float]]:
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @abstractmethod
    def supports_model(self, model_name: str) -> bool:
        pass
