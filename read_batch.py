import anthropic
import os
from dotenv import load_dotenv
import asyncio
import argparse

async def main(batch_id):
    load_dotenv()
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    result_stream = await client.beta.messages.batches.results(batch_id)
    async for entry in result_stream:
        if entry.result.type == "succeeded":
            print(entry.result.message.content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_id", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.batch_id))
