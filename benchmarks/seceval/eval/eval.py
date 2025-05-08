"""
SecEval Evaluation Script

This script evaluates language models on cybersecurity multiple-choice questions.
It supports various LLM backends including HuggingFace, Azure OpenAI, TextGen, and Ollama.
The script processes questions in batches and calculates accuracy scores by topic.

Usage:
    python3 eval.py -d dataset.json -B backend_type -m model_name [options]
Example:
    python3 eval.py --dataset_file datasets/questions.json --output_dir outputs --backend ollama --model ollama/qwen2.5:14b

Environment Variables:
    - OPENAI_API_ENDPOINT: Azure OpenAI endpoint URL
    - OPENAI_API_KEY: Azure OpenAI API key
    - TEXTGEN_MODEL_URL: URL for TextGen model
    - LOCAL_HF_MODEL_DIR: Directory containing local HuggingFace models
    - OLLAMA_API_BASE: Base URL for Ollama API (default: http://localhost:8000)
"""

import argparse
from typing import Any, Dict, List
from dotenv import load_dotenv
import asyncio
load_dotenv()
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chat_models import AzureChatOpenAI
from langchain.llms.textgen import TextGen
from langchain.schema.language_model import BaseLanguageModel
from langchain.adapters.openai import convert_message_to_dict
from langchain.schema.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage,
)
import json
import re
from pathlib import Path
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)
import time
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
import litellm

# Set up caching for LLM responses
set_llm_cache(
    SQLiteCache(
        database_path=str(Path(__file__).parent.parent / ".langchain.db")
    )
)

# Configure logging settings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger.addHandler(logging.FileHandler(f"./eval-{int(time.time())}.log", "w"))

# Instruction template for the LLM
instruction = """Below are multiple-choice questions concerning cybersecurity. 
Please select the correct answers and respond with the correct letters A, B, C, or D.
You could select more than one letter.
"""


def init_hf_llm(model_id: str) -> HuggingFacePipeline:
    """
    Initialize a HuggingFace language model.
    
    Args:
        model_id: The model identifier from HuggingFace
        
    Returns:
        HuggingFacePipeline: Initialized model pipeline
        
    Raises:
        ImportError: If required dependencies are not installed
    """
    # Check transformers and torch installation
    try:
        import transformers
    except ImportError:
        raise ImportError("Please install transformers with `pip install transformers`")
    try:
        import torch
        flash_attn_enable = torch.cuda.get_device_capability()[0] >= 8
    except ImportError:
        raise ImportError("Please install torch with `pip install torch`")

    # Initialize HuggingFace pipeline with specified parameters
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 5},
        device=0,
        model_kwargs={"trust_remote_code": True, "torch_dtype": torch.bfloat16},
    )
    return llm


def init_textgen_llm(model_id: str) -> TextGen:
    """
    Initialize a TextGen language model.
    
    Args:
        model_id: The model identifier
        
    Returns:
        TextGen: Initialized model
        
    Raises:
        RuntimeError: If TEXTGEN_MODEL_URL is not set
    """
    # Check for required environment variable
    if os.environ.get("TEXTGEN_MODEL_URL") is None:
        raise RuntimeError("Please set TEXTGEN_MODEL_URL")
    llm = TextGen(model_url=os.environ["TEXTGEN_MODEL_URL"])  # type: ignore
    return llm


def init_azure_openai_llm(model_id: str) -> AzureChatOpenAI:
    """
    Initialize an Azure OpenAI language model.
    
    Args:
        model_id: The model identifier
        
    Returns:
        AzureChatOpenAI: Initialized model
        
    Raises:
        RuntimeError: If required environment variables are not set
    """
    if os.environ.get("OPENAI_API_ENDPOINT") is None:
        raise RuntimeError("Please set OPENAI_API_ENDPOINT")
    if os.environ.get("OPENAI_API_KEY") is None:
        raise RuntimeError("Please set OPENAI_API_KEY")
    
    # Configure Azure OpenAI parameters
    azure_params = {
        "model": model_id,
        "openai_api_base": os.environ["OPENAI_API_ENDPOINT"],
        "openai_api_key": os.environ["OPENAI_API_KEY"],
        "openai_api_type": os.environ.get("OPENAI_API_TYPE", "azure"),
        "openai_api_version": "2023-07-01-preview",
    }
    return AzureChatOpenAI(**azure_params)  # type: ignore


def init_ollama_llm(model_id: str) -> 'OllamaChat':
    """
    Initialize an Ollama language model.
    
    Args:
        model_id: The model identifier
        
    Returns:
        OllamaChat: Initialized model wrapper
    """
    class OllamaChat:
        async def abatch(self, prompts: List[str]) -> List[str]:
            responses = []
            for prompt in prompts:
                try:
                    ollama_api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:8000")
                    api_base = ollama_api_base.rstrip('/v1')
                    completion = litellm.completion(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        api_base=api_base,
                        custom_llm_provider="ollama"
                    )
                    if hasattr(completion, "choices") and completion.choices:
                        content = completion.choices[0].message.content
                        result = self.extract_answer(content)
                        if result:
                            responses.append(result)
                        else:
                            print("Incorrect answer format detected.")
                            responses.append("Error: No result parsed")
                except Exception as e:
                    logging.error(f"Ollama error: {e}")
                    responses.append(f"Error: {e}")
            return responses

        def extract_answer(self, text: str) -> str:
            match = re.findall(r"[A-D]", text.upper())
            return "".join(sorted(set(match))) if match else ""

    return OllamaChat()


def init_openrouter_llm(model_id: str):
    class OpenRouterChat:
        async def abatch(self, prompts: List[str]):
            responses = []
            for prompt in prompts:
                try:
                    api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1/chat/completions")
                    api_key = os.getenv("OPENROUTER_API_KEY")

                    if not api_key:
                        raise ValueError("OPENROUTER_API_KEY is not defined in the environment variables.")

                    completion = litellm.completion(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        api_base=api_base,
                        api_key=api_key,
                        custom_llm_provider="openrouter"
                    )

                    if hasattr(completion, "choices") and completion.choices:
                        content = completion.choices[0].message.content
                        result = self.extract_answer(content)
                        if result:
                            responses.append(result)
                        else:
                            print("Formato de respuesta incorrecto.")
                            responses.append("Error: No se pudo extraer resultado")
                except Exception as e:
                    logging.error(f"OpenRouter error: {e}")
                    responses.append(f"Error: {e}")
            return responses

        def extract_answer(self, text: str):
            match = re.findall(r"[A-D]", text.upper())
            return "".join(sorted(set(match))) if match else ""

    return OpenRouterChat()


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation dataset from JSON file.
    
    Args:
        dataset_path: Path to dataset file
        
    Returns:
        List[Dict[str, Any]]: Loaded dataset
    """
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset


async def batch_inference_dataset(llm: BaseLanguageModel, batch: List[Dict[str, Any]], chat: bool = False) -> List[Dict[str, Any]]:
    """
    Process a batch of questions through the language model.
    
    Args:
        llm: Language model to use
        batch: List of questions to process
        chat: Whether to use chat format
        
    Returns:
        List[Dict[str, Any]]: Processed results with scores
    """
    results = []
    llm_inputs = []
    for dataset_row in batch:
        question_text = (
            "Question: " + dataset_row["question"] + " ".join(dataset_row["choices"])
        )
        question_text = question_text.replace("\n", " ")
        if chat:
            llm_input = [SystemMessage(content=instruction)]
        else:
            llm_input = instruction + "\n"

        llm_inputs.append(llm_input)
    try:
        llm_outputs = await llm.abatch(llm_inputs)
    except Exception as e:
        logging.error(f"error in processing batch {e}")
        llm_outputs = [f"{e}" * len(llm_inputs)]
    for idx, llm_output in enumerate(llm_outputs):
        if type(llm_output) == AIMessage:
            llm_output: str = llm_output.content  # type: ignore
        if "Answer:" in llm_output:
            llm_output = llm_output.replace("Answer:", "")
        if chat:
            batch[idx]["llm_input"] = convert_message_to_dict(llm_inputs[idx])
        else:
            batch[idx]["llm_input"] = llm_inputs[idx]
        batch[idx]["llm_output"] = llm_output
        batch[idx]["llm_answer"] = "".join(
            sorted(list(set(re.findall(r"[A-D]", llm_output))))
        )
        batch[idx]["score"] = int(
            batch[idx]["llm_answer"].lower() == batch[idx]["answer"].lower()
        )
        logging.info(
            f'llm_output: {llm_output}, parsed answer: {batch[idx]["llm_answer"]}, answer: {batch[idx]["answer"]}'
        )

        print("Question:", batch[idx]["question"])
        print("Correct Answer:", batch[idx]["answer"])
        print("LLM Answer:", batch[idx]["llm_answer"])
        print("LLM Output:", llm_output)
        print("Score:", batch[idx]["score"])
        print("--------------------------------")

        results.append(batch[idx])
    return results


def inference_dataset(
    llm: BaseLanguageModel,
    dataset: List[Dict[str, Any]],
    batch_size: int = 1,
    chat: bool = False,
):
    # Prepare the batched inference
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    # Asynchronously process dataset in batches
    loop = asyncio.get_event_loop()
    batches = list(chunks(dataset, batch_size))
    results = []
    for idx, batch in enumerate(batches):
        logger.info(f"processing batch {idx+1}/{len(batches)}")
        results += loop.run_until_complete(batch_inference_dataset(llm, batch, chat))
    return results


def count_score_by_topic(dataset: List[Dict[str, Any]]):
    score_by_topic = {}
    total_score_by_topic = {}
    score = 0
    for dataset_row in dataset:
        for topic in dataset_row["topics"]:
            if topic not in score_by_topic:
                score_by_topic[topic] = 0
                total_score_by_topic[topic] = 0
            score_by_topic[topic] += dataset_row["score"]
            total_score_by_topic[topic] += 1
        score += dataset_row["score"]
    score_fraction = {
        k: f"{v}/{total_score_by_topic[k]}" for k, v in score_by_topic.items()
    }
    score_float = {
        k: round(100 * float(v) / float(total_score_by_topic[k]), 4)
        for k, v in score_by_topic.items()
    }
    score_float["Overall"] = round(100 * float(score) / float(len(dataset)), 4)
    score_fraction["Overall"] = f"{score}/{len(dataset)}"
    return score_fraction, score_float


def main():
    parser = argparse.ArgumentParser(description="SecEval Evaluation CLI")

    parser.add_argument("-o", "--output_dir", type=str, default="/tmp", help="Specify the output directory.")
    parser.add_argument("-d", "--dataset_file", type=str, required=True, help="Specify the dataset file to evaluate on.")
    parser.add_argument("-c", "--chat", action="store_true", default=False, help="Evaluate on chat model.")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Specify the batch size.")
    parser.add_argument("-B", "--backend", type=str, choices=["remote_hf", "azure", "textgen", "local_hf", "ollama", "openrouter"], required=True, help="Specify the llm type. remote_hf: remote huggingface model backed, azure: azure openai model, textgen: textgen backend, local_hf: local huggingface model backed, ollama: ollama model, openrouter: openrouter model")
    parser.add_argument("-m", "--models", type=str, nargs="+", required=True, help="Specify the models.")
   
    args = parser.parse_args()
    models = list(args.models)

    logging.info(f"Evaluating models: {models}")
    for model_id in models:
        if args.backend == "remote_hf":
            llm = init_hf_llm(model_id)
        elif args.backend == "local_hf":
            model_dir = os.environ.get("LOCAL_HF_MODEL_DIR")
            if model_dir is None:
                raise RuntimeError(
                    "Please set LOCAL_HF_MODEL_DIR when using local_hf backend"
                )
            model_id = os.path.join(model_dir, model_id)
            llm = init_hf_llm(model_id)
        elif args.backend == "textgen":
            llm = init_textgen_llm(model_id)
        elif args.backend == "azure":
            llm = init_azure_openai_llm(model_id)
        elif args.backend == "ollama":
            llm = init_ollama_llm(model_id)
        elif args.backend == "openrouter":
            llm = init_openrouter_llm(model_id)
        else:
            raise RuntimeError("Unknown backend")

        dataset = load_dataset(args.dataset_file)
        result = inference_dataset(llm, dataset, batch_size=args.batch_size, chat=args.chat)
        score_fraction, score_float = count_score_by_topic(result)
        
        result_with_score = {
            "score_fraction": score_fraction,
            "score_float": score_float,
            "detail": result,
        }
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(args.dataset_file).stem}_{os.path.basename(model_id)}.json"
        
        logger.info(f"Writing result to {output_path}")
        with open(output_path, "w") as f:
            json.dump(result_with_score, f, indent=4)
        del llm

if __name__ == "__main__":
    main()