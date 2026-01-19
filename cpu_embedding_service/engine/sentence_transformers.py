from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer

from cpu_embedding_service.engine.base import EmbeddingEngine

logger = logging.getLogger(__name__)


class SentenceTransformerEngine(EmbeddingEngine):
    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None
        self._model_name: Optional[str] = None
        self._dimension: int = 0
        self._loaded_models: Dict[str, SentenceTransformer] = {}

    def load_model(self, model_name: str) -> None:
        if model_name in self._loaded_models:
            self._model = self._loaded_models[model_name]
            self._model_name = model_name
            self._dimension = self._model.get_sentence_embedding_dimension()
            return

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        
        try:
            self._model = SentenceTransformer(
                model_name,
                device="cpu",
                trust_remote_code=False
            )
            self._model_name = model_name
            self._dimension = self._model.get_sentence_embedding_dimension()
            self._loaded_models[model_name] = self._model
            logger.info(
                "Model loaded successfully",
                extra={
                    "model": model_name,
                    "dimension": self._dimension
                }
            )
        except Exception as e:
            logger.error(
                "Failed to load model",
                extra={
                    "model": model_name,
                    "error_type": type(e).__name__
                }
            )
            raise

    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        truncate: bool = True
    ) -> List[List[float]]:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )

        return embeddings.tolist()

    def get_dimension(self) -> int:
        return self._dimension

    def get_model_name(self) -> str:
        return self._model_name or ""

    def is_loaded(self) -> bool:
        return self._model is not None

    def supports_model(self, model_name: str) -> bool:
        return True
