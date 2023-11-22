from torch.utils.data import Dataset
import time
import os
from tqdm import tqdm
import numpy as np
import torch

from fastspeech2.text import text_to_sequence
from utils import process_text


def get_data_to_buffer(data_path, mel_ground_truth, aligment_path, text_cleaners, pitch_path, energy_path, batch_expand_size):
    buffer = []
    text = process_text(data_path)

    start = time.perf_counter()
    for i in tqdm(range(len(text)), "Loading dataset"):
        mel_gt_name = os.path.join(
            mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)

        duration = np.load(os.path.join(
            aligment_path, str(i)+".npy"))
        
        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, text_cleaners))
        
        pitch_gt_name = os.path.join(
            pitch_path, "ljspeech-pitch-%05d.npy" % (i+1))
        pitch_gt_target = np.load(pitch_gt_name).astype(np.float32)

        energy_gt_name = os.path.join(
            energy_path, "ljspeech-energy-%05d.npy" % (i+1))
        energy_gt_target = np.load(energy_gt_name).astype(np.float32)

        character = torch.from_numpy(character)
        mel_gt_target = torch.from_numpy(mel_gt_target)
        duration = torch.from_numpy(duration)
        pitch_gt_target = torch.from_numpy(pitch_gt_target)
        energy_gt_target = torch.from_numpy(energy_gt_target)

        buffer.append({
            "text": character,
            "duration": duration,
            "mel_target": mel_gt_target,
            "pitch": pitch_gt_target,
            "energy": energy_gt_target,
            "batch_expand_size": batch_expand_size
        })
    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


class LJSpeechDataset(Dataset):
    def __init__(self, data_path, mel_ground_truth, aligment_path, text_cleaners, pitch_path, energy_path, batch_expand_size, limit=None):
        self.buffer = get_data_to_buffer(data_path, mel_ground_truth, aligment_path, text_cleaners, pitch_path, energy_path, batch_expand_size)

        if limit is not None:
            self.buffer = self.buffer[:limit]
    
    def __getitem__(self, index):
        return self.buffer[index]
    
    def __len__(self):
        return len(self.buffer)