import os
import logging
import discord
import asyncio
import time
import math

from ollama import AsyncClient
ollama_client = AsyncClient(host='http://ollama_server:11434')

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')

class OllamaClient(discord.Client):

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

        # Ignore message not starting with !prompt command
        if not message.content.startswith("!prompt"):
            return

        prompt = message.content[8:]
        await message.channel.send(f"Processing prompt: {prompt}")

        response_obj = await ollama_client.chat(model='llama2-uncensored:7b-chat', messages=[{'role': 'user', 'content': prompt}])
        response = response_obj['message']['content']

        response_length = len(response)
        max_response_length = 2000

        if response_length < max_response_length:
            await message.channel.send(response)
            return

        response_chunk_count = math.ceil(response_length / max_response_length)

        for z in range(response_chunk_count):

            start_index = z * max_response_length
            end_index = (z + 1) * max_response_length

            if end_index > response_length:
                end_index = response_length

            await message.channel.send(response[start_index:end_index])

intents = discord.Intents.default()
intents.message_content = True

client = OllamaClient(intents = intents)
client.run(os.getenv('TOKEN'), log_handler = handler)
