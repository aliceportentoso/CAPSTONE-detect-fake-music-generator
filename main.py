import argparse
import os
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import sys

original_sys_path = sys.path.copy()
temp_path = os.path.join(os.path.dirname(__file__), 'lp-music-caps')
sys.path.append(temp_path)

from lpmc.music_captioning.model.bart import BartCaptionModel
from lpmc.utils.eval_utils import load_pretrained
from lpmc.utils.audio_utils import load_audio, STR_CH_FIRST
from omegaconf import OmegaConf

current_dir = os.path.dirname(__file__)
dataset_path = os.path.join(current_dir, "FakeMusicCapsSubset")
output_path = os.path.join(current_dir, "Results")

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument("--framework", default="transfer", type=str)
parser.add_argument("--caption_type", default="lp_music_caps", type=str)
parser.add_argument("--max_length", default=128, type=int)
parser.add_argument("--num_beams", default=5, type=int)
parser.add_argument("--model_type", default="last", type=str)
parser.add_argument("--audio_path", default="", type=str)


def get_audio(audio_path, duration=10, target_sr=16000):
    n_samples = int(duration * target_sr)
    audio, sr = load_audio(
        path=audio_path,
        ch_format=STR_CH_FIRST,
        sample_rate=target_sr,
        downmix_to_mono=True,
    )
    if len(audio.shape) == 2:
        audio = audio.mean(0, False)  # to mono
    input_size = int(n_samples)
    if audio.shape[-1] < input_size:  # pad sequence
        pad = np.zeros(input_size)
        pad[: audio.shape[-1]] = audio
        audio = pad
    ceil = int(audio.shape[-1] // n_samples)
    audio = torch.from_numpy(np.stack(np.split(audio[:ceil * n_samples], ceil)).astype('float32'))
    return audio


def captioning(args, output_file):

    hp_dir = "./lp-music-caps/lpmc/music_captioning/exp/transfer/lp_music_caps/hparams.yaml"
    config = OmegaConf.load(hp_dir)

    model = BartCaptionModel(max_length=config.max_length)
    device = torch.device('cpu')
    last_dir = "./lp-music-caps/lpmc/music_captioning/exp/transfer/lp_music_caps/last.pth"
    checkpoint = torch.load(last_dir, map_location="cpu", weights_only=True)

    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    audio_tensor = get_audio(audio_path=args.audio_path)

    with torch.no_grad():
        output = model.generate(
            samples=audio_tensor,
            num_beams=args.num_beams,
        )
    inference = {}
    number_of_chunks = range(audio_tensor.shape[0])
    for chunk, text in zip(number_of_chunks, output):
        time = f"{chunk * 10}:00-{(chunk + 1) * 10}:00"
        item = {"text": text, "time": time}
        inference[chunk] = item

    with open(output_file, "w") as f:
        for chunk, item in inference.items():
            f.write(f"{item['time']}: {item['text']}\n")

def main():
    args = parser.parse_args()
    print("end parse")
    for subdir, dirs, files in os.walk(dataset_path):
        print("subdir, dirs, files")
        for file in files:
            print("|")
            if file.endswith(".wav"):
                audio_path = os.path.join(subdir, file)
                args.audio_path = audio_path
                relative_subdir = os.path.relpath(subdir, dataset_path)
                output_subdir = os.path.join(output_path, relative_subdir)
                os.makedirs(output_subdir, exist_ok=True)
                output_file = os.path.join(output_subdir, file.replace(".wav", ".txt"))
                captioning(args, output_file)

if __name__ == '__main__':
    main()