import torch

from .base import BaseEmbedding
from transformers import AutoTokenizer, AutoModel

class BERTEmbedder(BaseEmbedding):
    def __init__(self,
                 model_name: str = '',
                 device: str = 'cpu',
                 normalize:bool = True,
                 max_length:int = 512
                 ):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.normalize = normalize

    def embed_text(self, text):
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        ).to(self.device)
        with torch.no_grad():
            output = self.model(**encoded)
    
        embedding = output.last_hidden_state[:, 0, :]

        if self.normalize:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.squeeze(0).cpu().numpy()

