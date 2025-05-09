"""
Benchmark Evaluation Script

This script provides utilities to evaluate language models on cybersecurity-related multiple-choice and other question-answering benchmarks.

Usage:
    python benchmarks/eval.py --model MODEL_NAME --dataset_file INPUT_FILE --eval EVAL_TYPE --backend BACKEND

Arguments:
    -m, --model           Specify the model to evaluate (e.g., "gpt-4", "qwen2.5:14b", etc.)
    -d, --dataset_file    Path to the dataset file (JSON or TSV) containing questions to evaluate
    -o, --output_dir      Specify the output directory for results (default: "benchmarks/outputs/[benchmark_name]")
    -B, --backend         Backend to use: "openai", "openrouter", "ollama" (required)
    -e, --eval            Specify the evaluation benchmark

Example:

     python benchmarks/eval.py --model ollama/qwen2.5:14b --dataset_file benchmarks/cybermetric/CyberMetric-2-v1.json --eval cybermetric --backend ollama
     python benchmarks/eval.py --model ollama/qwen2.5:14b --dataset_file benchmarks/seceval/eval/datasets/questions-2.json --eval seceval --backend ollama
     python benchmarks/eval.py --model ollama/qwen2.5:14b --dataset_file benchmarks/cti_bench/data/cti-mcq1.tsv --eval cti_bench --backend ollama
     python benchmarks/eval.py --model ollama/qwen2.5:14b --dataset_file benchmarks/cti_bench/data/cti-ate2.tsv --eval cti_bench --backend ollama
     python benchmarks/eval.py --model ollama/qwen2.5:14b --dataset_file benchmarks/cti_bench/data/cti-rcm2.tsv --eval cti_bench --backend ollama
     python benchmarks/eval.py --model ollama/qwen2.5:14b --dataset_file benchmarks/cti_bench/data/cti-vsp2.tsv --eval cti_bench --backend ollama
     python benchmarks/eval.py --model qwen/qwen3-32b:free --dataset_file benchmarks/cybermetric/CyberMetric-2-v1.json --eval cybermetric --backend openrouter

Environment Variables:
    OPENROUTER_API_KEY:  API key for OpenRouter (if using OpenRouter models)
    OPENROUTER_API_BASE: Base URL for OpenRouter API (default: https://openrouter.ai/api/v1)
    OLLAMA_API_BASE:     Base URL for Ollama API via LiteLLM proxy (default: http://localhost:8000/v1)
    OPENAI_API_KEY:      API key for OpenAI (if using OpenAI models)
    OPENAI_API_BASE:     Base URL for OpenAI API (default: https://api.openai.com/v1)
"""

import json
import re
import time
import os
import datetime
import random
import string
import argparse
from tqdm import tqdm
import litellm
import requests
import csv
import os
import datetime


OPENROUTER_API_BASE = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:8000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")


def ask_model(question_obj, instruction, model, api_base, api_key=None, custom_llm_provider=None):
    """
    Calls the model with the question and choices, returns the extracted answer.
    """
    # Compose the prompt
    prompt =  question_obj["Question"] + "\n" + instruction
    if question_obj.get("Choices"):
        prompt += "\nChoices:\n" + question_obj["Choices"]
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a security expert who answers questions."},
                {"role": "user", "content": prompt},
            ],
            api_base=api_base,
            api_key=api_key,
            custom_llm_provider=custom_llm_provider,
        )
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content
  
            return content
    except Exception as e:
        print(f"Error: {e}.")
        return None

def load_dataset(dataset_file, eval_type):
    questions = [] #list of questions: {question: str, answers: dict, solution: str}
    if eval_type == "seceval":
        with open(dataset_file, 'r') as f:
            data = json.load(f)
            for question in data:
                questions.append({
                    "Question": question["question"],
                    "Choices": "\n".join(question["choices"]),
                    "Solution": question["answer"]
                })

    elif eval_type == "cybermetric":
        with open(dataset_file, 'r') as f:
            data = json.load(f)
            for question in data.get("questions", []):
                questions.append({
                    "Question": question.get("question", ""),
                    "Choices": "\n".join([f"{k}: {v}" for k, v in question.get("answers", {}).items()]),
                    "Solution": question.get("solution", "")
                })
    elif eval_type == "cti_bench":
        with open(dataset_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            for row in reader:
                # Handle three possible formats:
                # Format 1: [URL, Question, Option A, Option B, Option C, Option D, Prompt, GT] (8 columns)
                # Format 2: [URL, Platform, Description, Prompt, GT] (5 columns)
                # Format 3: [URL, Description, Prompt, GT] (4 columns)

                if len(row) == 8:
                    # MCQ format
                    questions.append({
                        "Question": row[1],
                        "Choices": f"A: {row[2]}\nB: {row[3]}\nC: {row[4]}\nD: {row[5]}",
                        "Solution": row[7]
                    })
                elif len(row) == 5:
                    # ATE format (no choices, just open-ended)
                    questions.append({
                        "Question": row[2] + row[3],  # Description + Prompt
                        "Choices": "",       # No choices for ATE
                        "Solution": row[4]   # GT
                    })
                elif len(row) == 4:
                    # RCM format: [URL, Description, Prompt, GT]
                    questions.append({
                        "Question": row[1] + row[2],  # Description + Prompt
                        "Choices": "",       # No choices for RCM
                        "Solution": row[3]   # GT
                    })

    return questions

def run_evaluation(dataset, instruction, model, api_base=None, api_key=None, custom_llm_provider=None):
    results = []
    for idx, q in enumerate(dataset):
        model_answer = ask_model(q, instruction, model, api_base, api_key, custom_llm_provider)
        print(f"---------------{idx+1}/{len(dataset)}----------------")
        print(f"Evaluating question: {q['Question']}")
        print(f"Choices: {q['Choices']}")
        print(f"Solution: {q['Solution']}") 
        print(f"Model Answer: {model_answer}")
        print("--------------------------------")
        results.append({
            "Question": q["Question"],
            "Choices": q["Choices"],
            "ModelAnswer": model_answer,
            "Solution": q["Solution"]
        })
        
    return results

def compute_accuracy(results, benchmark_name, dataset_file=None):
    """
    Compute accuracy for a benchmark result set.

    Args:
        results (list of dict): Each dict should have the ground truth answer and model answer.
        benchmark_name (str): The name of the benchmark.
    Returns:
        accuracy (float): Accuracy as a percentage (0-100).
        correct_count (int): Number of correct answers.
        total_count (int): Total number of evaluated items.
    """
    correct_count = 0
    total_count = 0
    for item in results:
        sol = item.get("Solution")
        pred = item.get("ModelAnswer")
        # For cybermetric, parse both gt and pred using parse_result_cybermetric
        if benchmark_name.lower() == "cybermetric":
            from benchmarks.eval import parse_result_cybermetric
            pred_parsed = parse_result_cybermetric(pred)
            if sol is not None and pred_parsed is not None:
                if sol == pred_parsed:
                    correct_count += 1
                total_count += 1
        elif benchmark_name.lower() == "seceval":  
            pred_parsed = parse_result_seceval(pred)
            if sol is not None and pred_parsed is not None:
                if sol == pred_parsed:
                    correct_count += 1
                total_count += 1
        elif benchmark_name.lower() == "cti_bench":
            pred_parsed = parse_result_cti_bench(pred, dataset_file)
            print(pred_parsed)
            print(sol)
            if sol is not None and pred_parsed is not None:
                if sol == pred_parsed:
                    correct_count += 1
                total_count += 1
        else:
            if sol is not None and pred is not None:
                # Accept either exact match or case-insensitive match
                if str(sol).strip().upper() == str(pred).strip().upper():
                    correct_count += 1
                total_count += 1
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
    return accuracy, correct_count, total_count

def parse_result_seceval(result):
    # Expecting format: 'ANSWER: X', 'ANSWER: XY', or 'ANSWER: XYZ' (1, 2, or 3 letters A-D)
    if result is None:
        return None
    match = re.search(r"ANSWER:?\s*([A-D]{1,3})", result, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def parse_result_cybermetric(result):
    # Expecting format: 'ANSWER: X'
    if result is None:
        return None
    match = re.search(r"ANSWER:?\s*([A-D])", result, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def parse_result_cti_bench(result, dataset_file):
    # Accepts answers like 'ANSWER: X', 'ANSWER: XY', or 'ANSWER: XYZ' (A-D), and also 'ANSWER: Txxxx' (MITRE ATT&CK IDs)
    if result is None: 
        return None
    # Try to match CWE IDs (e.g., CWE-416, CWE-79, etc.)
    if "cti-rcm" in dataset_file:
        cwe_match = re.search(r"(CWE-\d+)", result, re.IGNORECASE)
        if cwe_match:
            return cwe_match.group(1).upper()
    # Try to match A-D answers (1-3 letters)
    if "cti-mcq" in dataset_file:
        match = re.search(r"ANSWER:?\s*([A-D])", result, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    # Find all MITRE ATT&CK technique IDs (e.g., T1071, T1059, etc.) anywhere in the result
    if "cti-ate" in dataset_file:
        ids = re.findall(r"T\d{4}", result, re.IGNORECASE)
        if ids:
            # Return all found IDs as uppercase, comma-separated
            return ", ".join([x.upper() for x in ids])
    if "cti-vsp" in dataset_file:
        # Try to match CVSS vector strings (e.g., CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H)
        cvss_match = re.search(r"(CVSS:3\.1/AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH])", result, re.IGNORECASE)
        if cvss_match:
            return cvss_match.group(1).upper()
        return result.upper()
    return None

def save_benchmark_results(
    benchmark_name,
    model,
    dataset_file,
    start_time,
    end_time,
    questions_processed,
    correct_count,
    accuracy,
    total_count,
    result
):
    """
    Save benchmark results in CyberMetric-style format to output_dir/information.txt.
    """
    output_dir = os.path.join(os.getcwd(), "benchmarks", "outputs", benchmark_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save information file as: <model>_<YYYYMMDD_HHMMSS>.txt
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = "".join([c if c.isalnum() or c in ('-', '_') else '_' for c in str(model)])
    info_file = os.path.join(output_dir, f"{safe_model}_{now_str}.txt")
    duration = end_time - start_time

    # Create a subdirectory for this run, named after info_file (without extension)
    run_dir = os.path.splitext(info_file)[0]
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Save the info file as .tct inside the run_dir
    info_file="information.txt"
    info_file_tct = os.path.join(run_dir, os.path.basename(os.path.splitext(info_file)[0] + ".txt"))
    with open(info_file_tct, "w") as f:
        f.write(f"{benchmark_name} Evaluation\n")
        f.write("=====================\n\n")
        f.write(f"Model: {model}\n")
        f.write(f"Dataset: {os.path.basename(dataset_file)}\n")
        f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Questions Processed: {questions_processed}\n")
        f.write(f"Correct Answers: {correct_count}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration}\n")

    # Save the results as answers.json inside the run_dir
    results_file = os.path.join(run_dir, "answers.json")
    with open(results_file, "w", encoding="utf-8") as rf:
        json.dump(result, rf, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SecEval Evaluation CLI")
    parser.add_argument("-d", "--dataset_file", type=str, required=True, help="Specify the dataset file to evaluate on.")
    parser.add_argument("-B", "--backend", type=str, choices=["openai", "ollama", "openrouter"], required=True, help="Specify the llm type. openai: openai model, ollama: ollama model, openrouter: openrouter model")
    parser.add_argument("-m", "--model", type=str, required=True, help="Specify the models.")
    parser.add_argument("-e", "--eval", type=str, required=True, help="Specify the evaluation benchmark.")
    args = parser.parse_args()
    
    model = args.model

    print(f"Evaluating model: {model}")
 
    if args.backend == "openai":
        api_base=OPENAI_API_BASE
        api_key=OPENAI_API_KEY
        custom_llm_provider=None
    elif args.backend == "ollama":
        api_base=OLLAMA_API_BASE
        api_base = api_base.rstrip('/v1')
        api_key=None
        custom_llm_provider="ollama"
    elif args.backend == "openrouter":
        api_base=OPENROUTER_API_BASE
        api_key=OPENROUTER_API_KEY
        custom_llm_provider="openrouter"
    else:
        raise RuntimeError("Unknown backend")

    if args.eval == "seceval":
        instruction = """Below are multiple-choice questions concerning cybersecurity. 
        Please select the correct answers and respond with the correct letters A, B, C, or D.
        You could select more than one letter
        Always return in this format: 'ANSWER: XYZ' or 'ANSWER: X'.
        """
        if args.dataset_file == "":
            args.dataset_file = "benchmarks/seceval/eval/datasets/questions-2.json"
        dataset = load_dataset(args.dataset_file, "seceval")
    elif args.eval == "cybermetric":
        instruction = """Choose the correct answer (A, B, C, or D) only. 
        Always return in this format: 'ANSWER: X'.
        """
        if args.dataset_file == "":
            args.dataset_file = "benchmarks/cybermetric/CyberMetric-2-v1.json"
        dataset = load_dataset(args.dataset_file, "cybermetric")
    elif args.eval == "cti_bench":
        if "ate" in args.dataset_file:
            instruction = """
            Return your answer in the MITRE ATT&CK technique ID format.
            You could select more than one technique ID. Be brief and concise. 
            Return your answer in this format: 'ANSWER: TXXXX' or 'ANSWER: TXXXX, TXXXX, TXXXX'
            Remember they are just an example, the important is the format."""
        elif "rcm" in args.dataset_file:
           instruction = """ #CWE
           Return CWE ID. 
           Return your answer in this format: 'ANSWER: CWE-XXXX' (X are numbers)."""   
        elif "mcq" in args.dataset_file:
            instruction = """
            Select the correct answer (A, B, C, or D) only.
            Return your answer in this format: 'ANSWER: A'"""
        elif "vsp" in args.dataset_file:
            instruction = """
            Return your answer in this CVSS format: 'ANSWER: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H'"""
        if args.dataset_file == "":
            args.dataset_file = "benchmarks/cti_bench/data/cti-mcq1.tsv"
        dataset = load_dataset(args.dataset_file, "cti_bench")
    
    start_time = datetime.datetime.now()
    result = run_evaluation(dataset, instruction, model, api_base, api_key, custom_llm_provider)
    end_time = datetime.datetime.now()

    print(result)

    accuracy, correct_count, total_count = compute_accuracy(result, args.eval, dataset_file=args.dataset_file)
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
    
    save_benchmark_results(args.eval, model, args.dataset_file, start_time, end_time, len(dataset), correct_count, accuracy, total_count, result)

if __name__ == "__main__":
    main()
