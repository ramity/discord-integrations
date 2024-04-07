import os
import logging
import discord
import asyncio
import time

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

# Load model.
# unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
# unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
# pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
unet = UNet2DConditionModel.from_config(base, subfolder="unet")
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float32, variant="fp16")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')

class TTIClient(discord.Client):

    async def on_ready(self):

        # Print some verbose messages to console
        print('Discord client ready. Logged on as ' + str(self.user))

    async def on_message(self, message):

        # Ignore messages sent by the bot itself to prevent recursion
        if message.author == self.user:
            return

        # Ignore messages not sent to the correct channel id
        if str(message.channel.id) != str(os.getenv('CHANNEL_ID')):
            return

        await message.channel.send(f"Processing prompt: {message.content}")

        # Run the pipeline, return the generated image, remove the file
        temp_filename = str(int(time.time())) + '.jpg'
        pipe(message.content, num_inference_steps = 4, guidance_scale = 0).images[0].save(temp_filename)
        await message.channel.send(file = discord.File(temp_filename))
        os.remove(temp_filename)

intents = discord.Intents.default()
intents.message_content = True

client = TTIClient(intents = intents)
client.run(os.getenv('TOKEN'), log_handler = handler)
