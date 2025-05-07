import json
import re
import time
from tqdm import tqdm
import os
import litellm


class CyberMetricEvaluator:
    def __init__(self, model_name, file_path):
        self.model_name = model_name  # E.g., "ollama/llama3"
        self.file_path = file_path
        self.report_file = "report_failed_questions.json"
        self.failed_questions = []
        self.failed_count = 0
        print("--DEBUG: model_name: ", self.model_name)

    def read_json_file(self):
        with open(self.file_path, 'r') as file:
            return json.load(file)

    def save_failed_questions(self):
        # Create the report with the same format as original file
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

    def ask_llm(self, question, answers, max_retries=5):
        options = ', '.join([f"{key}) {value}" for key, value in answers.items()])
        prompt = (
            f"Question: {question}\n"
            f"Options: {options}\n\n"
            "Choose the correct answer (A, B, C, or D) only. Always return in this format: 'ANSWER: X'"
        )
        print("--DEBUG: prompt: ", prompt)
        for attempt in range(max_retries):
            try:
                response = litellm.completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a security expert who answers questions."},
                        {"role": "user", "content": prompt},
                    ],
                    api_base="http://localhost:8000"  #"http://localhost:11434"  # Ollama API endpoint
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

    def run_evaluation(self):
        json_data = self.read_json_file()
        questions_data = json_data['questions']

        correct_count = 0
        incorrect_answers = []

        with tqdm(total=len(questions_data), desc="Processing Questions") as progress_bar:
            for item in questions_data:
                question = item['question']
                answers = item['answers']
                correct_answer = item['solution']

                llm_answer = self.ask_llm(question, answers)
                if llm_answer == correct_answer:
                    correct_count += 1
                else:
                    # Add the question to our failed questions list
                    self.failed_questions.append({
                        'question': question,
                        'answers': answers,
                        'solution': correct_answer,
                        'llm_answer': llm_answer
                    })
                    self.failed_count += 1
                    
                    # Save failed questions to JSON file every 2 failures
                    if self.failed_count % 2 == 0:
                        self.save_failed_questions()
                    
                    incorrect_answers.append({
                        'question': question,
                        'correct_answer': correct_answer,
                        'llm_answer': llm_answer
                    })

                accuracy_rate = correct_count / (progress_bar.n + 1) * 100
                progress_bar.set_postfix_str(f"Accuracy: {accuracy_rate:.2f}%")
                progress_bar.update(1)

        print(f"\nFinal Accuracy: {correct_count / len(questions_data) * 100:.2f}%")
        
        # Final save of failed questions
        if self.failed_questions:
            self.save_failed_questions()

        if incorrect_answers:
            print("\nIncorrect Answers:")
            for item in incorrect_answers:
                print(f"Question: {item['question']}")
                print(f"Expected Answer: {item['correct_answer']}, LLM Answer: {item['llm_answer']}\n")

if __name__ == "__main__":
    # Enable debug logging for litellm
    #litellm._turn_on_debug()
    
    file_path = 'CyberMetric-10-v1.json' #small set for testing
    
    # Use the exact model name as it appears in Ollama
    # For Ollama models, you should use "ollama/model_name"
    # The "ollama/" prefix tells litellm to use Ollama
    evaluator = CyberMetricEvaluator(model_name="ollama/qwen2.5:14b", file_path=file_path) # ollama/qwen3:32b-q8_0-ctx-32768"
    evaluator.run_evaluation()
