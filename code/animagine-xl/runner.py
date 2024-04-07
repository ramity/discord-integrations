import os
import logging
import discord
import asyncio
import time

import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-3.1", 
    torch_dtype=torch.float32, 
    use_safetensors=True, 
)

negative_prompt = "lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"

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
        image = pipe(message.content, negative_prompt=negative_prompt, width=1024, height=1024, guidance_scale=5, num_inference_steps=15).images[0].save(temp_filename)
        await message.channel.send(file = discord.File(temp_filename, spoiler=True))
        os.remove(temp_filename)

intents = discord.Intents.default()
intents.message_content = True

client = TTIClient(intents = intents)
client.run(os.getenv('TOKEN'), log_handler = handler)
