from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time


@dataclass
class DocumentMetadata:
    id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None # optional textual content stored with metadata
    created_at: float = field(default_factory=lambda: time.time())