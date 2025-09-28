from argparse import Namespace
from typing import Self
from pathlib import Path

import torch

class CFG(object):
    _instance: Self | None = None

    """Singleton Configuration Class"""
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self,
                 args: Namespace,
                 current_dir:Path,
                 data_dir:Path,
                 batch_size: int = 128,
                 num_workers: int = 0,
                 num_epochs: int = 300,
                 num_classes: int = 10,
                 device: torch.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
                 ) -> None:
        # For singleton pattern
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.args = args
        self.current_dir = current_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.device = device

    @classmethod
    def get_config(cls) -> Self:
        if cls._instance is not None:
            return cls._instance
        else:
            raise ValueError("Configuration has not been initialized yet.")