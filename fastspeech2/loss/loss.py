import torch
from torch import nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, 
                mel_predicted, 
                duration_predicted, 
                pitch_predicted, 
                energy_predicted, 
                mel_target, 
                duration, 
                pitch, 
                energy,
                *args, 
                **kwargs):
        mel_loss = self.mse_loss(mel_predicted, mel_target)

        duration_predictor_loss = self.mse_loss(duration_predicted,
                                               torch.log(duration.float() + 1))
        
        pitch_predictor_loss = self.mse_loss(pitch_predicted,
                                               torch.log(pitch.float() + 1))
        
        energy_predictor_loss = self.mse_loss(energy_predicted,
                                               torch.log(energy.float() + 1))

        return mel_loss, duration_predictor_loss, pitch_predictor_loss, energy_predictor_loss