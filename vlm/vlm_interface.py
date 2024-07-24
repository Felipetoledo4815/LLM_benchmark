from typing import List, Tuple
from abc import ABC, abstractmethod

class VLMInterface(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """Initializes the VLM"""

    @abstractmethod
    def inference(self, prompts: List[str], images: List[str]) -> Tuple[str, float]:
        """Returns the output produced by the VLM"""
