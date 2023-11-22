import os
import numpy as np
import librosa
import pyworld as pw
from scipy.interpolate import interp1d
from tqdm import tqdm

import audio
from audio.hparams_audio import hop_length

def generate_features(path: str):
    wav_path = os.path.join(path, 'LJSpeech-1.1/wavs')
    energy_path = os.path.join(path, 'energy')
    pitch_path = os.path.join(path, 'pitch')


    min_energy = None
    max_energy = None
    min_pitch = None
    max_pitch = None



    for i, filename in tqdm(enumerate(os.listdir(wav_path))):
        print(wav_path, filename)
        mel_spectrogram, energy = audio.tools.get_mel(os.path.join(wav_path, filename))
        mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)

        wav, sr = librosa.load(os.path.join(wav_path, filename))

        pitch, t = pw.dio(
            wav.astype(np.float64),
            sr,
            frame_period=hop_length / sr * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr)

        nonzero_idx = np.where(pitch != 0)[0]
        pitch = interp1d(
                nonzero_idx,
                pitch[nonzero_idx],
                fill_value=(pitch[nonzero_idx[0]], pitch[nonzero_idx[-1]]),
                bounds_error=False,
            )(np.arange(0, pitch.shape[0]))

        min_energy = energy.min() if min_energy is None else min(min_energy, energy.min())
        max_energy = energy.max() if max_energy is None else max(max_energy, energy.max())
        min_pitch = pitch.min() if min_pitch is None else min(min_pitch, pitch.min())
        max_pitch = pitch.max() if max_pitch is None else max(max_pitch, pitch.max())

        np.save(os.path.join(energy_path, str(i)), energy)
        np.save(os.path.join(pitch_path, str(i)), pitch)

    return {
        'min_energy': min_energy,
        'max_energy': max_energy,
        'min_pitch': min_pitch,
        'max_pitch': max_pitch,
    }

    