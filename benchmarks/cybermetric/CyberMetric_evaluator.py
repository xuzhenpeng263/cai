"""
CyberMetric Evaluator for LLMs

This script evaluates the performance of language models on the CyberMetric benchmark.
It supports both OpenRouter-hosted models and local Ollama models via LiteLLM proxy.

Usage:
    python CyberMetric_evaluator.py --model_name MODEL_NAME [--file_path FILE_PATH] [--api_key API_KEY]

Arguments:
    --model_name: Required. Model name with prefix (openrouter/ or ollama/)
                  Examples: openrouter/anthropic/claude-3-opus, ollama/llama3
    --file_path:  Optional. Path to the CyberMetric JSON file (default: CyberMetric-2-v1.json)
    --api_key:    Optional. API key for OpenRouter (can also use OPENROUTER_API_KEY env var)

Environment Variables:
    OPENROUTER_API_KEY:  API key for OpenRouter (if using OpenRouter models)
    OPENROUTER_API_BASE: Base URL for OpenRouter API (default: https://openrouter.ai/api/v1)
    OLLAMA_API_BASE:     Base URL for Ollama API via LiteLLM proxy (default: http://localhost:8000/v1)

Examples:
    # Run with an OpenRouter model
    python CyberMetric_evaluator.py --model_name openrouter/qwen/qwen3-32b:free

    # Run with a local Ollama model (requires LiteLLM proxy running)
    python CyberMetric_evaluator.py --model_name ollama/qwen2.5:14b

    # Specify a different benchmark file
    python CyberMetric_evaluator.py --model_name openrouter/qwen/qwen3-32b:free --file_path CyberMetric-10000-v1.json


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

# Enable debug mode for litellm
#litellm._turn_on_debug()

# Default API bases
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OLLAMA_LITELLM_API_BASE = os.environ["OLLAMA_API_BASE"] # "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "test_key_for_ci_environment"

class CyberMetricEvaluator:
    def __init__(self, model_name, file_path, api_key=None, openrouter_api_base=None, ollama_litellm_api_base=None):
        self.model_name = model_name  
        self.file_path = file_path
        self.failed_questions = []
        self.failed_count = 0
        
        # Set API configurations
        self.openrouter_api_base = openrouter_api_base or os.environ.get("OPENROUTER_API_BASE", OPENROUTER_API_BASE)
        self.ollama_litellm_api_base = ollama_litellm_api_base or os.environ.get("OLLAMA_LITELLM_API_BASE", OLLAMA_LITELLM_API_BASE)
        
        # Set API key for OpenRouter if needed
        self.api_key = None
        if self.model_name.startswith("openrouter/"):
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

        self.start_time = datetime.datetime.now()
        
        # Create output directory structure
        self.output_dir = self.create_output_directory()
        self.info_file = os.path.join(self.output_dir, "information.txt")
        self.report_file = os.path.join(self.output_dir, "report_failed_questions.json")
        
        # Initialize info file
        self.initialize_info_file()
        
        print("--DEBUG: model_name: ", self.model_name)
        
    def create_output_directory(self):
        # Create base directory if it doesn't exist
        base_dir = "outputs"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        # Format model name for directory (replace / with -)
        model_dir_name = self.model_name.replace("/", "-")
        
        # Get current date
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Create directory name
        dir_name = f"{model_dir_name}-{current_date}"
        
        # If directory already exists, add random string
        full_path = os.path.join(base_dir, dir_name)
        if os.path.exists(full_path):
            random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
            dir_name = f"{model_dir_name}-{current_date}-{random_str}"
            full_path = os.path.join(base_dir, dir_name)
        
        os.makedirs(full_path)
        return full_path
        
    def initialize_info_file(self):
        with open(self.info_file, 'w') as file:
            file.write(f"CyberMetric Evaluation\n")
            file.write(f"=====================\n\n")
            file.write(f"Model: {self.model_name}\n")
            file.write(f"Dataset: {self.file_path}\n")
            file.write(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Status: Running\n")
            file.write(f"Questions Processed: 0\n")
            file.write(f"Correct Answers: 0\n")
            file.write(f"Accuracy: 0.00%\n")

    def update_info_file(self, questions_processed, correct_count, status="Running"):
        accuracy = correct_count / questions_processed * 100 if questions_processed > 0 else 0
        
        with open(self.info_file, 'w') as file:
            file.write(f"CyberMetric Evaluation\n")
            file.write(f"=====================\n\n")
            file.write(f"Model: {self.model_name}\n") # "openrouter/qwen/qwen3-32b:free"
            file.write(f"Dataset: {self.file_path}\n")
            file.write(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Status: {status}\n")
            file.write(f"Questions Processed: {questions_processed}\n")
            file.write(f"Correct Answers: {correct_count}\n")
            file.write(f"Accuracy: {accuracy:.2f}%\n")
            
            if status == "Completed":
                end_time = datetime.datetime.now()
                duration = end_time - self.start_time
                file.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                file.write(f"Duration: {duration}\n")

    def read_json_file(self):
        with open(self.file_path, 'r') as file:
            return json.load(file)

    def save_failed_questions(self):
        report_data = {"questions": self.failed_questions}
        
        with open(self.report_file, 'w') as file:
            json.dump(report_data, file, indent=4)
        
        print(f"Updated failed questions report in {self.report_file}")

    @staticmethod
    def extract_answer(response):
        if response.strip():
            match = re.search(r"ANSWER:?\s*([A-D])", response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None
    
    def ask_ollama_litellm(self, prompt, max_retries=5):
        for attempt in range(max_retries):
            try:
                response = litellm.completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a security expert who answers questions."},
                        {"role": "user", "content": prompt},
                    ],
                    api_base="http://localhost:8000"
                )
                if hasattr(response, "choices") and response.choices:
                    content = response.choices[0].message.content
                    result = self.extract_answer(content)
                    if result:
                        print("--DEBUG: result: ", result)
                        return result
                    else:
                        print("Incorrect answer format detected. Attempting the question again.")
            except Exception as e:
                print(f"Error: {e}. Attempting the question again in {2 ** attempt} seconds.")
                time.sleep(2 ** attempt)
        return None

    def ask_openrouter(self, prompt, max_retries=5):
        if not self.api_key:
            raise ValueError("API key is required for OpenRouter models")
        
            
        for attempt in range(max_retries):
            try:
                response = litellm.completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a security expert who answers questions."},
                        {"role": "user", "content": prompt},
                    ],
                    api_base=self.openrouter_api_base,
                    api_key=self.api_key,
                    headers={
                        "HTTP-Referer": "https://your-site-url.com",  
                        "X-Title": "CyberMetric Evaluator"
                    }
                )
                if hasattr(response, "choices") and response.choices:
                    content = response.choices[0].message.content
                    result = self.extract_answer(content)
                    if result:
                        print("--DEBUG: result: ", result)
                        return result
                    else:
                        print("Incorrect answer format detected. Attempting the question again.")
            except Exception as e:
                print(f"Error: {e}. Attempting the question again in {2 ** attempt} seconds.")
                time.sleep(2 ** attempt)
        return None

    def ask_llm(self, question, answers, max_retries=5):
        options = ', '.join([f"{key}) {value}" for key, value in answers.items()])
        prompt = (
            f"Question: {question}\n"
            f"Options: {options}\n\n"
            "Choose the correct answer (A, B, C, or D) only. Always return in this format: 'ANSWER: X'"
        )
        print("--DEBUG: prompt: ", prompt)
        
        if self.model_name.startswith("openrouter/"):
            return self.ask_openrouter(prompt, max_retries)
        elif self.model_name.startswith("ollama/"):
            # Only use the litellm approach with port 8000 for Ollama
            return self.ask_ollama_litellm(prompt, max_retries)
        else:
            print("Error: Model name must start with 'openrouter/' or 'ollama/'")
            return None

    def run_evaluation(self):
        if not (self.model_name.startswith("openrouter/") or self.model_name.startswith("ollama/")):
            print("Error: You must set model name with prefix 'ollama/' or 'openrouter/'")
            return
            
        json_data = self.read_json_file()
        questions_data = json_data['questions']

        correct_count = 0
        incorrect_answers = []

        with tqdm(total=len(questions_data), desc="Processing Questions") as progress_bar:
            for i, item in enumerate(questions_data):
                question = item['question']
                answers = item['answers']
                correct_answer = item['solution']

                llm_answer = self.ask_llm(question, answers)
                if llm_answer == correct_answer:
                    correct_count += 1
                else:
                    self.failed_questions.append({
                        'question': question,
                        'answers': answers,
                        'solution': correct_answer,
                        'llm_answer': llm_answer
                    })
                    self.failed_count += 1
                    
                    if self.failed_count % 2 == 0:
                        self.save_failed_questions()
                    
                    incorrect_answers.append({
                        'question': question,
                        'correct_answer': correct_answer,
                        'llm_answer': llm_answer
                    })

                # Update progress and information file
                questions_processed = i + 1
                accuracy_rate = correct_count / questions_processed * 100
                progress_bar.set_postfix_str(f"Accuracy: {accuracy_rate:.2f}%")
                progress_bar.update(1)
                
                # Update info file every 5 questions
                if questions_processed % 5 == 0 or questions_processed == len(questions_data):
                    self.update_info_file(questions_processed, correct_count)

        # Final update with completed status
        self.update_info_file(len(questions_data), correct_count, "Completed")
        print(f"\nFinal Accuracy: {correct_count / len(questions_data) * 100:.2f}%")
        
        if self.failed_questions:
            self.save_failed_questions() # final failed questions

        if incorrect_answers:
            print("\nIncorrect Answers:")
            for item in incorrect_answers:
                print(f"Question: {item['question']}")
                print(f"Expected Answer: {item['correct_answer']}, LLM Answer: {item['llm_answer']}\n")

if __name__ == "__main__":
    litellm._turn_on_debug()
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='CyberMetric Evaluator for LLMs')
    parser.add_argument('--model_name', type=str, required=True, 
                        help='Model name with prefix (openrouter/ or ollama/)')
    parser.add_argument('--file_path', type=str, default='CyberMetric-2-v1.json',
                        help='Path to the CyberMetric JSON file')
    parser.add_argument('--api_key', type=str, 
                        help='API key for OpenRouter (can also use OPENROUTER_API_KEY env var)')
    
    args = parser.parse_args()
    
    model_name = args.model_name
    file_path = args.file_path
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    
    if model_name.startswith("ollama/"):
        # Ollama configuration
        evaluator = CyberMetricEvaluator(
            model_name=model_name,
            file_path=file_path
        )
        print(f"Using Ollama configuration with LiteLLM proxy on port 8000")
    
    elif model_name.startswith("openrouter/"):
        # OpenRouter configuration
        if not api_key:
            raise ValueError("API key must be provided via --api_key or OPENROUTER_API_KEY environment variable for OpenRouter models")
            
        evaluator = CyberMetricEvaluator(
            model_name=model_name,
            file_path=file_path,
            api_key=api_key,
            openrouter_api_base=os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        )
        print("Using OpenRouter configuration")
        
    else:
        raise ValueError("Model name must start with 'ollama/' or 'openrouter/'")
    
    # Run the evaluation
    evaluator.run_evaluation()

