"""
Functions for calling LLMs through OpenRouter API.
Mostly vibe-coded with Claude Sonnet 4. 
"""
import os
import hashlib
import json
import base64
from io import BytesIO
import ipdb
import atexit
import asyncio
from typing import List, Optional, Dict, Any, Union, Tuple
import lmdb
import httpx
from PIL import Image
from openai import OpenAI, AsyncOpenAI
from filelock import FileLock
import threading
import sys


class BatchProgressCounter:
    """Thread-safe progress counter for batch LLM operations."""

    def __init__(self, total_requests: int):
        self.total_requests = total_requests
        self.completed = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = threading.Lock()
        self._display_line = None

    def update_cache_hit(self):
        """Record a cache hit and update display."""
        with self.lock:
            self.completed += 1
            self.cache_hits += 1
            self._update_display()

    def update_cache_miss(self):
        """Record a cache miss and update display."""
        with self.lock:
            self.completed += 1
            self.cache_misses += 1
            self._update_display()

    def _update_display(self):
        """Update the progress display on the same line."""
        progress_text = (
            f"\rProgress: {self.completed}/{self.total_requests} completed "
            f"| Cache: {self.cache_hits} hits, {self.cache_misses} misses")
        print(progress_text, end="", flush=True)

    def finalize(self):
        """Print final newline to complete the progress display."""
        print()  # New line after progress is complete


class LLMCache:
    """LMDB-based cache for LLM responses."""

    def __init__(self, cache_dir: str = "./cache"):
        """Initialize the cache with LMDB."""
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, "llm_cache.lmdb")
        print(self.cache_path)
        self.lock_path = self.cache_path + ".lock"
        self.lock = FileLock(self.lock_path)
        self.env = lmdb.open(self.cache_path,
                             map_size=1024 * 1024 * 1024)  # 1GB max

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically close the cache."""
        self.close()

    def _generate_key(self, text: str, images: List[Image.Image],
                      model_name: str, max_tokens: int,
                      temperature: float) -> str:
        """Generate a unique cache key based on input parameters."""
        # Create a hash of the text, model, and generation parameters
        content = f"{text}|{model_name}|{max_tokens}|{temperature}"

        # Add image hashes if present
        if images:
            image_hashes = []
            for img in images:
                # Convert image to bytes and hash
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG')
                img_hash = hashlib.md5(img_bytes.getvalue()).hexdigest()
                image_hashes.append(img_hash)
            content += "|" + "|".join(image_hashes)

        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, images: List[Image.Image], model_name: str,
            max_tokens: int,
            temperature: float) -> Optional[Tuple[str, float]]:
        """Retrieve cached response and cost if it exists."""
        key = self._generate_key(text, images, model_name, max_tokens,
                                 temperature)

        with self.env.begin() as txn:
            cached_value = txn.get(key.encode())
            if cached_value:
                # Handle batch counter for cache hit
                global _batch_counter
                if _batch_counter:
                    _batch_counter.update_cache_hit()
                else:
                    # print("✓", end="", flush=True)
                    pass

                try:
                    # Try to parse as JSON (new format with cost)
                    cached_data = json.loads(cached_value.decode())
                    if isinstance(cached_data,
                                  dict) and "response" in cached_data:
                        return cached_data["response"], cached_data.get(
                            "cost", 0.0)
                    else:
                        # Fallback for old format (just response text)
                        return cached_data, 0.0
                except json.JSONDecodeError:
                    # Fallback for old format (just response text)
                    return cached_value.decode(), 0.0
        return None

    def set(self,
            text: str,
            images: List[Image.Image],
            model_name: str,
            max_tokens: int,
            temperature: float,
            response: str,
            cost: float = 0.0):
        """Cache the response and cost."""
        key = self._generate_key(text, images, model_name, max_tokens,
                                 temperature)

        # Store as JSON with both response and cost
        cache_data = {"response": response, "cost": cost}

        with self.lock:
            with self.env.begin(write=True) as txn:
                txn.put(key.encode(), json.dumps(cache_data).encode())

    def close(self):
        """Close the LMDB environment."""
        self.env.close()


# Global cache instance
_cache = LLMCache()

# Global batch counter for progress tracking
_batch_counter = None

# Register cleanup function to run when script exits
atexit.register(_cache.close)


def _encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    # Convert to RGB if necessary (for JPEG compatibility)
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def call_llm(text: str,
             images: Optional[List[Image.Image]] = None,
             model_name: str = "openai/gpt-4o-mini",
             provider: str = "openrouter",
             use_cache: bool = True,
             max_tokens: int = 1000,
             temperature: float = 0.7,
             include_cost: bool = False,
             json_mode: bool = False) -> Tuple[str, Optional[float]]:
    """
    Call an LLM through OpenRouter API with optional image inputs and caching.
    This is a synchronous function that uses asyncio internally for non-blocking I/O.
    
    Args:
        text: The text prompt to send to the LLM
        images: Optional list of PIL Images to include in the request
        model_name: The model to use (default: openai/gpt-4o-mini)
        provider: The provider to use (currently only supports openrouter)
        use_cache: Whether to use caching (default: True)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        include_cost: Whether to include cost information in response (default: False)
        json_mode: Whether to apply JSON mode for GPT models (default: False)
        
    Returns:
        Tuple of (response_text, cost_in_usd). If include_cost is False, cost will be None.
        
    Raises:
        ValueError: If provider is not supported or API key is missing
        Exception: For API call failures
    """
    if provider != "openrouter":
        raise ValueError("Currently only 'openrouter' provider is supported")

    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    # Normalize images input
    if images is None:
        images = []

    # Check cache first
    if use_cache:
        cached_result = _cache.get(text, images, model_name, max_tokens,
                                   temperature)
        if cached_result:
            cached_response, cached_cost = cached_result
            if include_cost:
                return cached_response, cached_cost
            else:
                return cached_response, None

    # If we reach here, it's either a cache miss or cache is disabled
    global _batch_counter
    if _batch_counter:
        _batch_counter.update_cache_miss()
    else:
        print("✗", end="", flush=True)  # Original behavior for single calls

    async def _async_call():
        # Initialize OpenAI client with OpenRouter endpoint
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=30.0,
            max_retries=2,
            http_client=httpx.AsyncClient(
                verify=False
            )  # Disable SSL verification to fix certificate issues
        )

        # Prepare the message content
        message_content = [{"type": "text", "text": text}]

        # Add images if provided
        for image in images:
            image_b64 = _encode_image_to_base64(image)
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_b64
                }
            })

        # Prepare request parameters
        request_params = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": message_content
            }],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        # Add JSON mode for GPT models if requested
        if json_mode and model_name.startswith("openai/gpt"):
            request_params["response_format"] = {"type": "json_object"}

        try:
            if include_cost:
                # Make direct HTTP request when cost tracking is needed
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/jmhb/PaperSearch-rl",
                    "X-Title": "PaperSearch-RL"
                }

                # Add usage parameter for cost tracking
                request_params["usage"] = {"include": True}

                async with httpx.AsyncClient(verify=False,
                                             timeout=30.0) as http_client:
                    http_response = await http_client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=request_params)

                    if http_response.status_code != 200:
                        raise Exception(
                            f"HTTP {http_response.status_code}: {http_response.text}"
                        )

                    response_data = http_response.json()
                    response_text = response_data["choices"][0]["message"][
                        "content"]
                    cost = response_data.get("usage", {}).get("cost", 0.0)

                    # Cache the response with cost
                    if use_cache and response_text:
                        _cache.set(text, images, model_name, max_tokens,
                                   temperature, response_text, cost)

                    return response_text, cost
            else:
                # Use OpenAI client for regular requests (without cost tracking)
                response = await client.chat.completions.create(
                    **request_params)
                response_text = response.choices[0].message.content

                # Cache the response with cost (0.0 for non-cost requests)
                if use_cache and response_text:
                    _cache.set(text, images, model_name, max_tokens,
                               temperature, response_text, 0.0)

                if include_cost:
                    return response_text, 0.0
                else:
                    return response_text, None

        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")

    # Run the internal async function from the synchronous wrapper
    try:
        return asyncio.run(_async_call())
    except KeyboardInterrupt:
        print("\nAPI call interrupted by user.")
        return "[ERROR: Interrupted]", None


async def _async_call_llm_batch_processor(prompts, images_list, model_name,
                                          provider, use_cache, max_tokens,
                                          temperature, max_concurrent,
                                          include_cost, json_mode):
    """The core async logic for batch processing."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _call_single(prompt, images):
        async with semaphore:
            try:
                # This needs to be a non-async call to the wrapper `call_llm`
                # which handles its own event loop via asyncio.run
                # To avoid loop-in-loop, we run it in a thread.
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, call_llm, prompt,
                                                    images, model_name,
                                                    provider, use_cache,
                                                    max_tokens, temperature,
                                                    include_cost, json_mode)
                return result
            except Exception as e:
                return e

    tasks = [
        _call_single(prompt, images)
        for prompt, images in zip(prompts, images_list)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    processed_costs = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            error_msg = f"Error for prompt {i}: {str(result)}"
            print(f"⚠️  {error_msg}")
            processed_results.append(f"[ERROR: {str(result)}]")
            if include_cost:
                processed_costs.append(0.0)
        else:
            response_text, cost = result
            processed_results.append(response_text)
            if include_cost:
                processed_costs.append(cost if cost is not None else 0.0)

    if include_cost:
        return processed_results, processed_costs
    return processed_results, None


def call_llm_batch(
        prompts: List[str],
        images_list: Optional[List[List[Image.Image]]] = None,
        model_name: str = "openai/gpt-4o-mini",
        provider: str = "openrouter",
        use_cache: bool = True,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        max_concurrent: int = 50,
        include_cost: bool = False,
        json_mode: bool = False) -> Tuple[List[str], Optional[List[float]]]:
    """
    Call LLM with multiple prompts concurrently using the existing call_llm function.
    This function is synchronous from the outside, but uses asyncio internally
    for concurrent, interruptible execution.
    
    Args:
        prompts: List of text prompts to send to the LLM
        images_list: Optional list of image lists (one per prompt)
        model_name: The model to use (default: openai/gpt-4o-mini)
        provider: The provider to use (currently only supports openrouter)
        use_cache: Whether to use caching (default: True)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        max_concurrent: Maximum number of concurrent requests (default: 50)
        include_cost: Whether to include cost information in response (default: False)
        json_mode: Whether to apply JSON mode for GPT models (default: False)
        
    Returns:
        Tuple of (responses_list, costs_list). If include_cost is False, costs_list will be None.
        
    Raises:
        ValueError: If inputs are invalid
        Exception: For API call failures
    """
    if not prompts:
        if include_cost:
            return [], []
        return [], None

    # Validate and normalize inputs
    if images_list is None:
        images_list = [[] for _ in prompts]
    elif len(images_list) != len(prompts):
        raise ValueError("images_list must have the same length as prompts")

    # Set up batch progress counter
    global _batch_counter
    _batch_counter = BatchProgressCounter(len(prompts))

    try:
        result = asyncio.run(
            _async_call_llm_batch_processor(prompts, images_list, model_name,
                                            provider, use_cache, max_tokens,
                                            temperature, max_concurrent,
                                            include_cost, json_mode))

        # Finalize progress display
        _batch_counter.finalize()
        return result

    except KeyboardInterrupt:
        # Intercept the KeyboardInterrupt from asyncio.run to provide a clear message.
        if _batch_counter:
            _batch_counter.finalize()
        print("🚫 Batch processing interrupted by user. Shutting down.")
        # Return empty lists to signal partial/interrupted completion.
        if include_cost:
            return [], []
        return [], None
    except Exception as e:
        if _batch_counter:
            _batch_counter.finalize()
        raise Exception(f"Batch processing failed: {str(e)}")
    finally:
        # Reset batch counter
        _batch_counter = None


def list_available_models() -> List[Dict[str, Any]]:
    """
    Get list of available models from OpenRouter.
    
    Returns:
        List of model dictionaries with model information
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    try:
        response = client.models.list()
        return [model.dict() for model in response.data]
    except Exception as e:
        raise Exception(f"Failed to fetch models: {str(e)}")


def test_llm_calls():
    """Test function to call LLM with different models."""
    prompt = "what does Steve Irwin say?"

    print("Testing LLM API calls...")
    print("=" * 50)

    # Test with GPT-4o-mini
    try:
        print(f"\n🤖 Testing with openai/gpt-4o-mini:")
        print(f"Prompt: {prompt}")
        response1, cost1 = call_llm(prompt,
                                    model_name="openai/gpt-4o-mini",
                                    use_cache=False)
        print(f"Response: {response1}")
        print(f"Cost: {cost1}")
    except Exception as e:
        print(f"❌ Error with gpt-4o-mini: {e}")

    print("\n" + "-" * 50)

    # Test with Qwen
    try:
        print(f"\n🤖 Testing with qwen/qwen-2.5-72b-instruct:")
        print(f"Prompt: {prompt}")
        response2, cost2 = call_llm(prompt,
                                    model_name="qwen/qwen-2.5-72b-instruct")
        print(f"Response: {response2}")
        print(f"Cost: {cost2}")
    except Exception as e:
        print(f"❌ Error with qwen: {e}")

    print("\n" + "=" * 50)
    print("Testing complete!")


def test_batch_llm_calls():
    """Test function for batch LLM calls."""
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.", "What does a cat say?",
        "Name three programming languages.", "What is 2 + 2?"
    ]

    print("Testing Batch LLM API calls...")
    print("=" * 50)
    print(f"Processing {len(prompts)} prompts concurrently...")

    try:
        import time
        start_time = time.time()

        responses, costs = call_llm_batch(
            prompts=prompts,
            model_name="openai/gpt-4o-mini",
            max_concurrent=2  # Limit to 3 concurrent requests
        )

        end_time = time.time()

        print(f"\n✅ Batch completed in {end_time - start_time:.2f} seconds")
        print("\nResults:")
        print("-" * 50)

        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            print(f"\n{i+1}. Prompt: {prompt}")
            print(f"   Response: {response}")

    except Exception as e:
        print(f"❌ Batch test failed: {e}")


def test_cost_tracking():
    """Test function to demonstrate cost tracking functionality."""
    print("Testing Cost Tracking...")
    print("=" * 50)

    # Test single call with cost tracking
    prompt = "What is the capital of France?"

    try:
        print(f"\n🤖 Testing single call with cost tracking:")
        print(f"Prompt: {prompt}")
        response, cost = call_llm(prompt,
                                  model_name="openai/gpt-4o-mini",
                                  include_cost=True,
                                  use_cache=False)
        print(f"Response: {response}")
        print(f"💰 Cost: ${cost:.8f} USD")
    except Exception as e:
        print(f"❌ Error with single call: {e}")

    print("\n" + "-" * 50)

    # Test batch calls with cost tracking
    prompts = ["What is 2 + 2?", "Name a color.", "What does a dog say?"]

    try:
        print(f"\n🤖 Testing batch calls with cost tracking:")
        print(f"Prompts: {prompts}")

        import time
        start_time = time.time()

        responses, costs = call_llm_batch(prompts=prompts,
                                          model_name="openai/gpt-4o-mini",
                                          include_cost=True,
                                          use_cache=False,
                                          max_concurrent=3)

        end_time = time.time()

        print(f"\n✅ Batch completed in {end_time - start_time:.2f} seconds")
        print("\nResults with costs:")
        print("-" * 50)

        total_cost = sum(costs)
        for i, (prompt, response,
                cost) in enumerate(zip(prompts, responses, costs)):
            print(f"\n{i+1}. Prompt: {prompt}")
            print(f"   Response: {response}")
            print(f"   💰 Cost: ${cost:.8f} USD")

        print(f"\n💰 Total Cost: ${total_cost:.8f} USD")

    except Exception as e:
        print(f"❌ Batch test failed: {e}")


# Example usage and testing
if __name__ == "__main__":
    # test_llm_calls()
    print("\n" + "=" * 70)
    test_batch_llm_calls()
    import sys
    sys.exit()
    print("\n" + "=" * 70)
    test_cost_tracking()

    # # Example of a single call
    # response, cost = call_llm("What is the capital of France?",
    #                           include_cost=True)
    # print(f"\nResponse: {response}")
    # if cost is not None:
    #     print(f"Cost: ${cost:.8f} USD")

    # Example of a batch call
    prompts = ["What is 2+2?"] * 10 + ["Name a color."] * 10
    responses, costs = call_llm_batch(prompts=prompts, include_cost=True)
    if costs:
        total_cost = sum(costs)
        print(f"\nTotal cost for batch: ${total_cost:.8f} USD")
        print("Batch responses:", responses)
