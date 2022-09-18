import torch.nn as nn
import torch.nn.functional as F
import torch

class Seq2seq(nn.Module):

    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        '''
        input_variable -->[B,Feat_Dim, Feats_Len]
        input_lengths --->[B,Feats_Len]
        target_variable ---> [B, Dec_T)]
        
        '''
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths) # [16 209 128] [4 16 128]
        decoder_outputs,sequence_symbols = self.decoder(inputs=target_variable, #[16 100]
                              encoder_hidden=encoder_hidden, # [4 16 128]
                              encoder_outputs=encoder_outputs, # [16 181 128]
                              function=self.decode_function, # F.log_softmax
                              teacher_forcing_ratio=teacher_forcing_ratio) # 0
        # decoder_outputs:list 100 [16 30],sequence_symbols: list 100 [16 1]
        
        final_dec_outputs = torch.stack(decoder_outputs,dim=2) # [16 30 100]
        final_sequence_symbols = torch.stack(sequence_symbols,dim=1).squeeze() # [16 100]
        return final_dec_outputs,final_sequence_symbols












