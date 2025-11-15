import re
import time
import torch
import numpy as np
from typing import List, Dict
from jiwer import wer
import psutil


class Timer:    
    def __init__(self, device="cuda"):
        self.device = device
        
    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def get_wer(references: List[str], hypotheses: List[str]) -> float:
    return wer(references, hypotheses) * 100


def get_latency(latencies: List[float]) -> Dict[str, float]:
    latencies_ms = np.array(latencies) * 1000 
    
    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "min_ms": float(np.min(latencies_ms)),
        "max_ms": float(np.max(latencies_ms)),
    }


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def extract_output(result):
    """    
    Output structure:
    [
        (
            '<eng><asr> text...',  # [0][0] - with prefix
            tokens,                # [0][1]
            token_ids,             # [0][2]
            'text...',             # [0][3] - clean text
            score                  # [0][4]
        )
    ]
    Args:
        result: Output from Speech2TextGreedySearch
    Returns:
        str: Clean transcription text
    """
    if isinstance(result, list) and len(result) > 0:
        f = result[0]
        if isinstance(f, tuple) and len(f) >= 4:
            # Index [3] is clean text
            return f[3]
        return str(f)
    return str(result)