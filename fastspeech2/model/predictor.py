import torch.nn as nn
import torch
import torch.nn.functional as F

from fastspeech2.model.utils import create_alignment


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)

class Predictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, encoder_dim, predictor_filter_size, predictor_kernel_size, dropout):
        super(Predictor, self).__init__()

        self.input_size = encoder_dim
        self.filter_size = predictor_filter_size
        self.kernel = predictor_kernel_size
        self.conv_output_size = predictor_filter_size
        self.dropout = dropout


        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out
    
class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = Predictor(encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout)
    
    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output
    
    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)
        if target is None:
            duration_predictor_output = ((torch.exp(duration_predictor_output) - 1) * alpha + 0.5).int()
            output = self.LR(x, duration_predictor_output)
            mel_pos = torch.stack([torch.Tensor([i + 1 for i in range(output.size(1))])]).long().to(x.device)
            return output, mel_pos
        else:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
