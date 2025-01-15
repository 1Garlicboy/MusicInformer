'''调用nn.transformer的enc_dec，不需要decoder的话直接去掉即可,之前没有maxpool'''

import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random

from utilities.constants import *
from utilities.device import get_device

from .positional_encoding import PositionalEncoding,RelativePositionalEncoding,RotationalEncoding,Fundamental_Music_Embedding,Music_PositionalEncoding
from .rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR,MyTransformerLayer
from encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from decoder import Decoder,DecoderLayer
from attn import FullAttention, ProbAttention,LocalAttention, AttentionLayer
from .rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR,MultiheadAttentionRPR,TransformerDecoder,TransformerDecoderLayer,TransformerEncoder,TransformerEncoderLayer,TransformerEncoderLayerSMA
from pdb import set_trace
import math

# MusicInformer
class MusicInformer(nn.Module):
    def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,e_layers=[3,2,1],
                 dropout=0.1, max_sequence=2048, rpr=False,attn='pro', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(MusicTransformer, self).__init__()

        self.dummy      = DummyDecoder()
        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.dropout2   =torch.nn.Dropout(dropout)
        self.max_seq    = max_sequence
        self.rpr        = rpr
        
        self.conv1 = nn.Conv1d(in_channels=d_model,
                                  out_channels=2*self.d_ff,
                                  kernel_size=11,
                                  padding=5,
                                  padding_mode='circular')
        self.conv2 = nn.Conv1d(in_channels=2*self.d_ff,
                                  out_channels=d_model,
                                  kernel_size=11,
                                  padding=5,
                                  padding_mode='circular')
        
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.d_ff, num_layers=2, batch_first=True)
    
        self.relu=nn.ReLU()
        # Input embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)


        #Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        factor=30
        output_attention=False
        d_ff=self.d_ff
        n_heads=self.nhead
        activation='relu' #informer默认为gelu，可调整
        e_layers=2 #informer
        distil=True
        mix=True

        encoder_layer1 = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)

        self.encoder = Encoder(

            [   
               
                EncoderLayer(
                    AttentionLayer(ProbAttention(False,factor=factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff, dropout=dropout,
                    activation=activation
                ),
                #  encoder_layer1,
                EncoderLayer(
                    AttentionLayer(ProbAttention(False,factor=factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff, dropout=dropout,
                    activation=activation
                ),
                # encoder_layer1,
                EncoderLayer(
                    AttentionLayer(ProbAttention(False,factor=factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff, dropout=dropout,
                    activation=activation
                ),
                encoder_layer1,
                encoder_layer1,
                encoder_layer1,
            ],
            [
                ConvLayer(
                    d_model,d_ff
                ) for l in range(n_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
            )
        
        # Final output is a softmaxed linear layer
        self.Wout=nn.Linear(self.d_model,VOCAB_SIZE)
        self.Wout1=nn.Linear(self.d_ff,VOCAB_SIZE)

        self.softmax=nn.Softmax(dim=-1)

    def forward(self, x,mask=True):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Takes an input sequence and outputs predictions using a sequence to sequence method.

        A prediction at one index is the "next" prediction given all information seen previously.
        ----------
        """

        if(mask is True):
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        else:
            mask = None

        tgt_mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        memory_mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())

        print(x.shape)


        x = self.embedding(x)
        x*=math.sqrt(self.d_ff)
        x = self.positional_encoding(x)
        x_out = self.encoder(x,attn_mask=mask)
        y,(h_n, c_n)=self.lstm(x_out)
        y=self.Wout1(y)

        del mask

        # They are trained to predict the next note in sequence (we don't need the last one)
        return y
        


    def generate(self, primer=None,target_seq_length=1024, beam=0, beam_chance=1.0):
        
        assert (not self.training), "Cannot generate while in training mode"

        print("Generating sequence of max length:", target_seq_length)

        gen_seq = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())

        
        cur_i=num_primer
        while(cur_i < target_seq_length):
            x=gen_seq[...,:cur_i]


            w=self.forward(x)
            x=self.softmax(w)
            y=x[..., :TOKEN_END]



            
            print('y_shape:',y.shape)
            print('curi-1:',cur_i-1)
            token_probs=y[:,cur_i-1,:]

            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)

            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)

                beam_rows = top_i // VOCAB_SIZE
                beam_cols = top_i % VOCAB_SIZE

                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols

            else:
                token_probs = token_probs[0]
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                print("next token:",next_token)
                gen_seq[:, cur_i] = next_token


                # Let the transformer decide to end if it wants to
                if(next_token == TOKEN_END):
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break

            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i]

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation_size, padding=padding)
            relu = nn.ReLU()
            dropout_layer = nn.Dropout(dropout)
            
            layers += [conv, relu, dropout_layer]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):

        return self.network(x)
    


class DummyDecoder(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    A dummy decoder that returns its input. Used to make the Pytorch transformer into a decoder-only
    architecture (stacked encoders with dummy decoder fits the bill)
    ----------
    """

    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask,tgt_key_padding_mask,memory_key_padding_mask):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Returns the input (memory)
        ----------
        """

        return memory
