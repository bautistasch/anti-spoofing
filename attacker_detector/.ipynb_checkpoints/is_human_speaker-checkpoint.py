import os
import json
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List
from typing import List, Tuple
from .models.utils import load_weights_from_pl_pipeline
from .models.model_builders import AudioClassificationModel

def is_human_speaker(filename):
    """
    Predice si es una persona o parlante

    Parameters:
    filename (str): path del archivo

    Return:
    bool: True. si es humano
    """
    def load_model(model_dir):
        model_config = json.loads((model_dir/"model_config.json").read_text())
        model = AudioClassificationModel(**model_config)
        model = model.eval()
        weights_path = str(model_dir/"model.ckpt")
        load_weights_from_pl_pipeline(model,str(weights_path),remove_unessacary=False,strict=False)
        return model

    current_dir = Path(__file__).resolve().parent
    model_dir = current_dir / "checkpoints" / "antispoofing" / "lrpd_office_lrpd_aparts"

    model = load_model(model_dir)
    predictions = []

    x, fs = sf.read(filename)
    x = torch.tensor(x, dtype=torch.float32)
    x = x.unsqueeze(0)
    x = x.unsqueeze(1)
    with torch.no_grad():
        a = model(x)
        pred = torch.nn.functional.softmax(a, dim=-1)
        predictions.append(pred.cpu().numpy())  

    predictions = np.concatenate(predictions)
    return predictions[0, 0] > 0.99
