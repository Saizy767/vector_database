import torch
import logging
from .base import BaseEmbedding
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = '', device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading SentenceTransformer model '{model_name}' on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def embed_text(self, text: str):
        encoded_input = self.tokenizer(text,
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt").to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
        logger.debug(f"Embedded text (len={len(text)}) → vector shape {embedding[0].shape}")
        return embedding[0].cpu().numpy()