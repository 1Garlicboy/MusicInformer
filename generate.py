import torch
import torch.nn as nn
import os
import random

from third_party.midi_processor.processor import decode_midi, encode_midi

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_informer import MusicInformer
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy, process_midi
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

from utilities.constants import *
from utilities.device import get_device, use_cuda

# Setting a random seed for reproducibility  
random.seed(42)  # You can change the number to any arbitrary integer

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """

    args = parse_generate_args()
    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    # Grabbing dataset if needed
    _, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, random_seq=False)

    # Sample list of all indices to track used primers  
    dataset_indices = list(range(len(dataset)))  
    random.shuffle(dataset_indices)  # 2. 随机打乱数据集索引

    midi_dir = '/home/sh/garlicboy/MusicTransformer-Pytorch-master/dataset/mae'  
    midi_files = [f for f in os.listdir(midi_dir) if f.endswith('.midi')]  

    # 对每个 MIDI 文件执行生成  
    for i, midi_filename in enumerate(midi_files):  
        try:  
            midi_path = os.path.join(midi_dir, midi_filename)  
            raw_mid = encode_midi(midi_path)  
            if len(raw_mid) == 0:  
                raise ValueError(f"No MIDI messages in primer file: {midi_filename}")  
            
            primer, _ = process_midi(raw_mid, args.num_prime, random_seq=False)  
            primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())  
        
            print(f"Using primer file: {midi_filename}")  
        
            model = MusicInformer(  
                n_layers=args.n_layers,  
                num_heads=args.num_heads,  
                d_model=args.d_model,  
                dim_feedforward=args.dim_feedforward,  
                max_sequence=args.max_sequence,  
                rpr=args.rpr  
            ).to(get_device())  
        
            model.load_state_dict(torch.load(args.model_weights))  
        
            # GENERATION 1个样本  
            model.eval()  
            with torch.set_grad_enabled(False):  
                if(args.beam > 0):  
                    print("BEAM:", args.beam)  
                    beam_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=args.beam)  
                    f_path = os.path.join(args.output_dir, f"beam_{i+1}.mid")  
                    decode_midi(beam_seq[0].cpu().numpy(), file_path=f_path)  
                else:  
                    print("RAND DIST")  
                    rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0)  
                    f_path = os.path.join(args.output_dir, f"rand_{i+1}.mid")  
                    decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)  

            print(f"Successfully generated {f_path}")  

        except Exception as e:  
            print(f"Error processing file {midi_filename}: {e}")  
            # Skip to the next MIDI file  
            continue  

if __name__ == "__main__":  
    main()
