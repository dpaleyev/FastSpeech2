import torch.nn as nn
import torch
import numpy as np

from fastspeech2.model.transformer_layers import Encoder, Decoder
from fastspeech2.model.predictor import LengthRegulator, Predictor
from fastspeech2.model.utils import get_mask_from_lengths


class FastSpeech2(nn.Module):

    def __init__(self,
                max_seq_len,
                vocab_size,
                num_encoder_layer,
                num_decoder_layer,
                encoder_dim,
                decoder_dim,
                encoder_head,
                decoder_head,
                encoder_conv1d_filter_size,
                decoder_conv1d_filter_size,
                fft_conv1d_kernel,
                fft_conv1d_padding,
                duration_predictor_filter_size,
                duration_predictor_kernel_size,
                pitch_predictor_filter_size,
                pitch_predictor_kernel_size,
                energy_predictor_filter_size,
                energy_predictor_kernel_size,
                min_pitch,
                max_pitch,
                min_energy,
                max_energy,
                num_bins,
                num_mels,
                PAD,
                dropout
                ):
        super(FastSpeech2, self).__init__()

        self.PAD = PAD

        self.encoder = Encoder(max_seq_len, 
            num_encoder_layer,
            vocab_size,
            encoder_dim,
            encoder_head,
            encoder_conv1d_filter_size,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            PAD,
            dropout
        )

        self.decoder = Decoder(max_seq_len,
            num_decoder_layer,
            vocab_size,
            decoder_dim,
            decoder_head,
            decoder_conv1d_filter_size,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            PAD,
            dropout
        )

        self.length_regulator = LengthRegulator(encoder_dim, 
            duration_predictor_filter_size,
            duration_predictor_kernel_size,
            dropout
        )

        self.register_buffer('pitch_bounds', torch.linspace(np.log(min_pitch + 1), np.log(max_pitch + 2), num_bins))
        self.pitch_embedding = nn.Embedding(num_bins, encoder_dim)
        self.pitch_predictor = Predictor(encoder_dim,
            pitch_predictor_filter_size,
            pitch_predictor_kernel_size,
            dropout
        )

        self.register_buffer('energy_bounds', torch.linspace(np.log(min_energy + 1), np.log(max_energy + 2), num_bins))
        self.energy_embedding = nn.Embedding(num_bins, encoder_dim)
        self.energy_predictor = Predictor(encoder_dim,
            energy_predictor_filter_size,
            energy_predictor_kernel_size,
            dropout
        )

        self.mel_linear = nn.Linear(decoder_dim, num_mels)
    
    def calc_pitch(self, encoder_output, pitch_target=None, beta=1.0):
        pitch_predictor_output = self.pitch_predictor(encoder_output)
        if self.training:
            embedding = self.pitch_embedding(torch.bucketize(torch.log(pitch_target + 1), self.pitch_bounds))
        else:
            energy_predictpitch_predictor_outpution_pred = pitch_predictor_output * beta
            embedding = self.pitch_embedding(torch.bucketize(torch.log(pitch_predictor_output), self.pitch_bounds))
        
        return embedding, pitch_predictor_output

    def calc_energy(self, encoder_output, energy_target=None, beta=1.0):
        energy_predictor_output = self.energy_predictor(encoder_output)
        if self.training:
            embedding = self.energy_embedding(torch.bucketize(torch.log(energy_target + 1), self.energy_bounds))
        else:
            energy_predictor_output = energy_predictor_output * beta
            embedding = self.energy_embedding(torch.bucketize(torch.log(energy_predictor_output), self.energy_bounds))
        
        return embedding, energy_predictor_output
    
    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)
    
    def forward(self, src_seq, 
                src_pos, 
                mel_pos, 
                mel_max_length=None, 
                length_target=None, 
                pitch_target=None, 
                energy_target=None,
                alpha=1.0, beta=1.0, gamma=1.0):
        
        x, _ = self.encoder(src_seq, src_pos)
        
        if self.training:
            output, duration_predictor_output = self.length_regulator(x, alpha, 
                                                            length_target, mel_max_length)

            pitch_emb, pitch_predictor_output = self.calc_pitch(output, 
                                                               pitch_target=pitch_target, beta=beta)

            energy_emb, energy_predictor_output = self.calc_energy(output, 
                                                            energy_target=energy_target, gamma=gamma)
            output = self.decoder(output + pitch_emb + energy_emb, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)

            return {"mel_output": output, 
                    "duration_predictor_output": duration_predictor_output,
                    "pitch_predictor_output": pitch_predictor_output,
                    "energy_predictor_output": energy_predictor_output}
        else:
            output, mel_pos = self.length_regulator(x, alpha)
            pitch_emb, _ = self.get_pitch(output)
            energy_emb, _ = self.get_energy(output)
            output = self.decoder(output + pitch_emb + energy_emb, mel_pos)
            output = self.mel_linear(output)
            return output

