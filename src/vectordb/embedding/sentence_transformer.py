import torch

from .base import BaseEmbedding
from transformers import AutoTokenizer, AutoModel
from typing import List


class SentenceTransformerEmbedding(BaseEmbedding):
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
        
    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        """
        Mean pooling, учитывающий маску внимания.
        """
        token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def embed_text(self, text: str):
        """
        Получение эмбеддинга для одной строки текста.
        """
        encoded_input = self.tokenizer(text,
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt").to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return embedding[0].cpu().numpy()