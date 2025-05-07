"python3 benchmarks/seceval/eval/eval_ollama.py --dataset_file benchmarks/seceval/eval/datasets/questions.json --output_dir benchmarks/seceval/eval/outputs --model ollama/qwen2.5:14b"
import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

from langchain.schema.messages import SystemMessage, HumanMessage
from langchain.adapters.openai import convert_message_to_dict
import litellm

load_dotenv()

def init_ollama_llm(model_id: str):
    class OllamaChat:
        async def abatch(self, prompts: List[str]):
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

        def extract_answer(self, text: str):
            match = re.findall(r"[A-D]", text.upper())
            return "".join(sorted(set(match))) if match else ""

    return OllamaChat()


def load_dataset(dataset_path: str):
    with open(dataset_path, "r") as f:
        return json.load(f)


async def batch_inference_dataset(llm, batch: List[Dict[str, Any]], chat=False):
    results = []
    llm_inputs = []
    instruction = """Below are multiple-choice questions concerning cybersecurity. 
    Please select the correct answer.
    Your exprected answer is JUST THE CORRECT LETTE, not a word or phrase."""

    for row in batch:
        question_text = "Question: " + row["question"] + " " + " ".join(row["choices"])
        question_text = question_text.replace("\n", " ")
        if chat:
            llm_input = [SystemMessage(content=instruction), HumanMessage(content=question_text)]
        else:
            llm_input = instruction + question_text + "\n"
        llm_inputs.append(llm_input)

    try:
        llm_outputs = await llm.abatch(llm_inputs)
    except Exception as e:
        logging.error(f"Error in batch: {e}")
        llm_outputs = [f"{e}"] * len(llm_inputs)

    for idx, output in enumerate(llm_outputs):
        output = output.replace("Answer:", "").strip()
        row = batch[idx]
        row["llm_input"] = llm_inputs[idx]
        row["llm_output"] = output
        row["llm_answer"] = "".join(sorted(set(re.findall(r"[A-D]", output))))
        row["score"] = int(row["llm_answer"].lower() == row["answer"].lower())
        results.append(row)

    return results


def inference_dataset(llm, dataset: List[Dict[str, Any]], batch_size: int = 1, chat=False):
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    loop = asyncio.get_event_loop()
    batches = list(chunks(dataset, batch_size))
    results = []
    for idx, batch in enumerate(batches):
        print(f"Processing batch {idx+1}/{len(batches)}")
        results += loop.run_until_complete(batch_inference_dataset(llm, batch, chat))
    return results


def count_score_by_topic(dataset: List[Dict[str, Any]]):
    score_by_topic = {}
    total_by_topic = {}
    total_score = 0

    for row in dataset:
        for topic in row["topics"]:
            score_by_topic[topic] = score_by_topic.get(topic, 0) + row["score"]
            total_by_topic[topic] = total_by_topic.get(topic, 0) + 1
        total_score += row["score"]

    fraction = {k: f"{score_by_topic[k]}/{total_by_topic[k]}" for k in score_by_topic}
    percent = {k: round(score_by_topic[k] / total_by_topic[k] * 100, 2) for k in score_by_topic}
    percent["Overall"] = round(total_score / len(dataset) * 100, 2)
    fraction["Overall"] = f"{total_score}/{len(dataset)}"

    return fraction, percent


def main():
    parser = argparse.ArgumentParser(description="Ollama LLM Benchmark")
    parser.add_argument("-d", "--dataset_file", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default="./results")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-c", "--chat", action="store_true", default=False)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_id = args.model
    llm = init_ollama_llm(model_id)

    dataset = load_dataset(args.dataset_file)
    results = inference_dataset(llm, dataset, batch_size=args.batch_size, chat=args.chat)
    score_frac, score_pct = count_score_by_topic(results)

    output = {
        "score_fraction": score_frac,
        "score_percent": score_pct,
        "details": results
    }

    output_path = Path(args.output_dir) / f"{Path(args.dataset_file).stem}_{model_id}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
