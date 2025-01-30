from datasets import load_dataset
from anthropic import AsyncAnthropic, AnthropicError, RateLimitError
import json
import time
from dotenv import load_dotenv
import os
import random
import argparse
import asyncio
from tqdm import tqdm
import uuid

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1
                print(f"Retrying {num_retries} times")

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def get_wmdp_dataset(subset=None):
    """Load the WMDP dataset using the datasets library"""
    try:
        dataset = load_dataset("cais/wmdp", subset)
        print("Successfully loaded WMDP dataset")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def format_prompt(question, choices):
    """Format the question and choices for Claude"""
    prompt = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{i}. {choice}\n"
    prompt += "Provide only the number (0-3) of the correct answer."
    return prompt


requests = []


@retry_with_exponential_backoff
async def get_model_response(model, messages, batch=False):
    """Get a response from the model"""
    if batch:
        print(f"Batch request appended")
        requests.append(
            {
                "custom_id": str(uuid.uuid4()),
                "params": {
                    "model": "claude-3-5-sonnet-latest",
                    "max_tokens": 1024,
                    "messages": messages,
                },
            }
        )
        return

    return await model.messages.create(
        model="claude-3-5-sonnet-latest",
        system="Provide only the number (0-3) of the correct answer.",
        max_tokens=1024,
        temperature=0,
        messages=messages,
    )


async def evaluate_model(
    dataset, other_dataset, model, num_fake_turns=0, use_prefixes=False, batch=False
):
    """Evaluate Claude on the dataset"""
    correct = 0
    total = 0
    skipped = 0

    for item in tqdm(list(dataset["test"])[:200]):
        prompt = format_prompt(item["question"], item["choices"])
        messages = []
        if num_fake_turns > 0:
            # pick n random qa pairs from the dataset that are not the same as the current item
            # and add them to the prompt
            if other_dataset:
                random_items = random.sample(
                    list(other_dataset["test"]), num_fake_turns
                )
            else:
                random_items = random.sample(list(dataset["test"]), num_fake_turns)
            for random_item in random_items:
                if random_item["question"] == item["question"]:
                    continue
                if not use_prefixes:
                    messages.append(
                        {
                            "role": "user",
                            "content": format_prompt(
                                random_item["question"], random_item["choices"]
                            ),
                        }
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"{random_item['answer']}",
                        }
                    )
                else:
                    print("using prefixes")
                    prompt += f"USER: {format_prompt(random_item['question'], random_item['choices'])}\nASSISTANT: {random_item['answer']}\n\n"

        messages.append({"role": "user", "content": prompt})
        response = await get_model_response(model, messages, batch)
        if not batch:
            try:
                prediction = int(response.content[0].text.strip())
                if prediction == item["answer"]:
                    correct += 1
                total += 1

                if total % 10 == 0:
                    print(
                        f"Progress: {total}/200 questions processed. Current accuracy: {(correct/total)*100:.2f}%"
                    )

            except ValueError:
                print(f"Skipping question (model refused to answer)")
                print(response.content[0].text)
                skipped += 1
                continue
    if batch:
        print(f"Batch request sent with {len(requests)} requests")
        batch_response = await model.beta.messages.batches.create(requests=requests)
        print(batch_response)
        return None

    print(f"\nSkipped {skipped} questions due to safety filters")
    return correct / total if total > 0 else 0


async def zip_async_gen_list(async_gen, lst):
    it = aiter(async_gen)  # Convert the async generator into an async iterator
    for item in lst:
        try:
            entry = await anext(it)  # Get the next item from the async generator
            yield entry, item
        except StopAsyncIteration:
            break  # Stop if the async generator is exhausted

async def evaluate_batch(dataset, model, batch_id):
    result_stream = await model.beta.messages.batches.results(batch_id)
    total = 0
    skipped = 0
    correct = 0
    async for entry, item in zip_async_gen_list(result_stream, list(dataset["test"])[:200]):
        if entry.result.type == "succeeded":
            total += 1
            try:
                prediction = int(entry.result.message.content[0].text.strip())
                if prediction == item["answer"]:
                    correct += 1
            except ValueError:
                skipped += 1
    print(f"Skipped {skipped} questions due to safety filters")
    return correct / total if total > 0 else 0

async def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_fake_turns",
        type=int,
        default=0,
        help="Number of fake conversation turns to add",
    )
    parser.add_argument(
        "--use_prefixes",
        action="store_true",
        help="Use prefixes instead of roles",
    )
    parser.add_argument(
        "--cross_task",
        action="store_true",
        help="Use cross-task datasets",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch requests",
    )
    parser.add_argument(
        "--batch_id",
        type=str,
        help="Batch ID to read",
    )

    args = parser.parse_args()
    client = AsyncAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )  # Replace with your API key
    wmdp_dataset = get_wmdp_dataset("wmdp-bio")
    if args.cross_task:
        other_dataset = get_wmdp_dataset("wmdp-cyber")
    else:
        other_dataset = None

    if args.batch_id:
        accuracy = await evaluate_batch(wmdp_dataset, client, args.batch_id)
        print(f"\nFinal accuracy: {accuracy*100:.2f}%")
    elif wmdp_dataset:
        print(f"Using prefixes: {args.use_prefixes}")
        accuracy = await evaluate_model(
            wmdp_dataset,
            other_dataset,
            client,
            args.num_fake_turns,
            args.use_prefixes,
            args.batch,
        )
        if accuracy:
            print(f"\nFinal accuracy: {accuracy*100:.2f}%")
        
        


if __name__ == "__main__":
    asyncio.run(main())
