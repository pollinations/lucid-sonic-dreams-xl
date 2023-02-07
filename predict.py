# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import sys
import os
import re
from typing import List, Optional, Tuple, Union
from importlib.machinery import SourceFileLoader

import click
import numpy as np
import PIL.Image
import torch

from imagenet_classes import nameToClassIdx 

sys.path.append("/content")

sys_path = list(sys.path)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.
    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m
    

models = [
    "imagenet (XL)",
    "ffhq (XL)",
    "pokemon (XL)"
]

super_resolution = False  # @param {type: "boolean"}
output_path = '/content'
model_map = {
    "imagenet (XL)": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet1024.pkl",
    "ffhq (XL)": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/ffhq1024.pkl",
    "pokemon (XL)": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon1024.pkl",
    # "humans": "/content/StyleGAN-Human/pretrained_models/stylegan2_1024.pkl",
}



class Predictor(BasePredictor):
    """Audio-To-Video - Lucid Sonic Dreams XL

    <img src="https://ipfs.pollinations.ai/ipfs/Qmc9D9poeppLMv7AMmrtH2VYgSdJnpwrgpuAgzzC7eMJ84" width="300" height="300" />

    *jellyfish*


    Upload an audio file to generate a music video that moves with every sound and produces abstract art by traveling through the latent space of a StyleGAN

    ---

    [Thomash](https://twitter.com/pollinations_ai) and [Niels](https://twitter.com/nielsrolf1) adapted [Lucid Sonic Dreams](https://github.com/mikaelalafriz/lucid-sonic-dreams) to use [StyleGAN-XL](https://sites.google.com/view/stylegan-xl/) and the [StyleGAN-human model](https://stylegan-human.github.io/)

    Lucid Sonic Dreams syncs GAN-generated visuals to music. By default, it uses [NVLabs StyleGAN2](https://github.com/NVlabs/stylegan2), with pre-trained models lifted from [Justin Pinkney's consolidated repository](https://github.com/justinpinkney/awesome-pretrained-stylegan2). Custom weights and other GAN architectures can be used as well.

    For a more detailed description of the technique refer to: [Introducing Lucid Sonic Dreams: Sync GAN Art to Music with a Few Lines of Python Code!](https://towardsdatascience.com/introducing-lucid-sonic-dreams-sync-gan-art-to-music-with-a-few-lines-of-python-code-b04f88722de1)

    Sample output can be found on [YouTube](https://youtu.be/l-nGC-ve7sI) and [Instagram](https://www.instagram.com/lucidsonicdreams/).

    **[UPD 17.10.2021]** Exposed more parameters
    [UPD 1.10.2021] Added Visionary Art Dataset
    """

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        model_type: str = Input(
            description="Which checkpoint to use?",
            # default="humans (StyleGAN-humans)",
            default="imagenet (XL)",
            choices=models,
        ),
        style: str = Input(
            description="The style to use. Only works for imagenet model",
            default="jellyfish",
            choices=list(nameToClassIdx.keys()),
        ),
        audio_file: Path = Input(
            description="Path to the uploaded audio file (.mp3, .wav are supported)"
        ),
        fps: int = Input(
            description="Frames per second of generated video", default=20
        ),
        pulse_react: int = Input(
            description="The 'strength' of the pulse. It is recommended to keep this between 0 and 100.",
            default=60,
        ),
        pulse_react_to: str = Input(
            description="Whether the pulse should react to percussive or harmonic elements",
            choices=["percussive", "harmonic"],
            default="percussive",
        ),
        motion_react: int = Input(
            description="The 'strength' of the motion. It is recommended to keep this between 0 and 100.",
            default=60,
        ),
        motion_react_to: str = Input(
            description="Whether the motion should react to percussive or harmonic elements",
            choices=["percussive", "harmonic"],
            default="harmonic",
        ),
        motion_randomness: int = Input(
            description="Degree of randomness of motion. Higher values will typically prevent the video from cycling through the same visuals repeatedly. Must range from 0 to 100.",
            default=50,
        ),
        truncation: int = Input(
            description='Controls the variety of visuals generated. Lower values lead to lower variety. Note: A very low value will usually lead to "jittery" visuals. Must range from 0 to 100.',
            default=50,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # sys.path = sys_path

        # os.system(f"ffmpeg -y -i {audio_file} /tmp/audio.wav")
        # os.system("ls -l /tmp")
        # audio_file = '/tmp/audio.wav'
        # os.system("echo hey > out.txt")

        classIdx = nameToClassIdx[style]
        if model_type == "humans (StyleGAN-humans)":
            network_pkl = "/content/StyleGAN-Human/pretrained_models/stylegan2_1024.pkl"
        else:
            network_pkl = f"/content/{model_type}.pkl"


        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        pulse_percussive = pulse_react_to == "percussive"
        pulse_harmonic = pulse_react_to == "harmonic"

        motion_percussive = motion_react_to == "percussive"
        motion_harmonic =  motion_react_to == "harmonic"


        
        # Import the correct version of LSD
        if model_type == "humans":
        #   %cd /content/StyleGAN-Human/
            repo_path = "/content/StyleGAN-Human/"
        else:
            repo_path = "/src/stylegan_xl/"

        os.chdir(repo_path)
        sys.path.append(repo_path)
        import dnnlib
        import legacy


        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

        noise_dim = G.z_dim

        def generate_images(
            G,
            z,
            truncation_psi: float,
            noise_mode: str,
            translate: Tuple[float,float],
            rotate: float,
            class_idx: Optional[int]
        ):
            """Generate images using pretrained network pickle.
            Examples:
            \b
            # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
            python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
                --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
            \b
            # Generate uncurated images with truncation using the MetFaces-U dataset
            python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
                --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
            """
            # Labels.
            label = torch.zeros([len(z), G.c_dim], device=device)
            if G.c_dim != 0:
                if class_idx is None:
                    raise click.ClickException('Must specify class label with --class when using a conditional network')
                label[:, class_idx] = 1
            else:
                if class_idx is not None:
                    print ('warn: --class=lbl ignored when running on an unconditional network')

            # Generate images.
            #for seed_idx, seed in enumerate(seeds):

            # Construct an inverse rotation/translation matrix and pass to the generator.  The
            # generator expects this matrix as an inverse to avoid potentially failing numerical
            # operations in the network.
            if hasattr(G.synthesis, 'input'):
                m = make_transform(translate, rotate)
                m = np.linalg.inv(m)
                G.synthesis.input.transform.copy_(torch.from_numpy(m))

            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return [PIL.Image.fromarray(i.cpu().numpy(), 'RGB') for i in img]
        
        def projected_gan(noise_batch, class_batch):
            noise_tensor = torch.from_numpy(noise_batch).cuda().float()
            return generate_images(G, noise_tensor, 1, "const", (0,0), 0, classIdx)

        import lucidsonicdreams as lsd

        L = lsd.LucidSonicDream(song = str(audio_file),
                            style = projected_gan, 
                            input_shape = noise_dim,
                            num_possible_classes = 0)



        L.hallucinate(file_name = 'output.mp4',
                    resolution = None,
                    fps = fps,
                    motion_percussive = motion_percussive,
                    motion_harmonic = motion_harmonic,
                    pulse_percussive = pulse_percussive,
                    pulse_harmonic = pulse_harmonic,
                    pulse_react = pulse_react / 100,
                    motion_react = motion_react / 100,
                    motion_randomness = motion_randomness / 100,
                    truncation = truncation / 100,
                    start = 0, 
                    batch_size = 8,
                    )
    

        if model_type == "humans":
            os.system(f'ffmpeg -y -i output.mp4 -vcodec libx264 -filter_complex "[0]pad=w=820:h=ih:x=153:y=0:color=black" {output_path}/lowres_output.mp4')
        else:
            os.system(f"cp -v output.mp4 {output_path}/lowres_output.mp4")
        
        try:
            del G
        except: # G is not defined if standard lsd is used
            pass
        del lsd
        del dnnlib
        return Path("/content/lowres_output.mp4")
