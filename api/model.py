import math
from typing import List

def predict_probs(features: List[float]) -> float:
    """
    Mock model for demonstation 
    """
    if not features:
        raise ValueError("features list is empty")
    
    score = sum(features) / len(features)
    return score
