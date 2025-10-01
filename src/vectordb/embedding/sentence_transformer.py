import torch

from .base import BaseEmbedding
from transformers import AutoTokenizer, AutoModel
from typing import List


class SentenсeTransformerEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = '', device: str = 'cpu'):
        """
        Args:
            model_name: название модели HuggingFace
            device: наименование девайса
        """

        self.model_name = model_name
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        

    @torch.no_grad()
    def embed_text(self, text: str) -> List[float]:
        """
        Генерирует эмбеддинг для одного текста
        """
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded_input)
            
        # Берем mean pooling по токенам
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].cpu().numpy()