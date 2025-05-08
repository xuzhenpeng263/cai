# AI Model Evaluation Benchmarks

This chapter is a curated collection of benchmark datasets and evaluation tools designed to assess the capabilities of custom AI models, particularly in domains related to cybersecurity.

The collection is intended to support researchers and developers who are evaluating their own models using reliable, task-specific benchmarks.

Currently, this are the benchmark included:

- [SecEval](https://github.com/XuanwuAI/SecEval)
- [CyberMetric](https://github.com/CyberMetric)

The goal is to consolidate diverse evaluation tasks under a single framework to support rigorous, standardized testing.

## 🏆 General Summary Table

| Model       | SecEval   | CyberMetric  | Total Value | 
|-------------|-----------|--------------|-------------|
| model_name  | `XX.X%`   | `XX.X%`      | `XX.X%`     | 



## 🔐 SecEval: [https://github.com/XuanwuAI/SecEval](https://github.com/XuanwuAI/SecEval)  

### 📄 Description

SecEval is a benchmark designed to evaluate large language models (LLMs) on security-related tasks. It includes various real-world scenarios such as phishing email analysis, vulnerability classification, and response generation.

### 📥 Installation

```bash
git clone https://github.com/XuanwuAI/SecEval.git
cd SecEval
pip install -r requirements.txt
```
### ▶️ Usage
```bash
python evaluate.py --model your_model_name --task all
```
### 📊 Evaluation Results

| Model Name     | Accuracy | F1 Score | ROUGE | Notes               |
|----------------|----------|----------|-------|---------------------|
| GPT-4          | 87.5%    | 84.2%    | 0.61  | Zero-shot           |
| LLaMA2-13B     | 75.4%    | 71.8%    | 0.52  | Fine-tuned          |
| Claude 3 Opus  | 79.2%    | 76.5%    | 0.58  | Few-shot setup      |
| Falcon-40B     | 70.1%    | 68.0%    | 0.47  | Baseline            |
| YourModel      | XX.X%    | XX.X%    | XX.X  | Custom results here |

📂 Source: results/seceval/scores.csv

---

## 🧠 CyberMetric: [https://github.com/CyberMetric](https://github.com/CyberMetric)  

### 📄 Description 
CyberMetric is a benchmark framework that focuses on measuring the performance of AI systems in cybersecurity-specific question answering, knowledge extraction, and contextual understanding. It emphasizes both domain knowledge and reasoning ability.

### 📥 Installation
```bash
git clone https://github.com/CyberMetric/CyberMetric.git
cd CyberMetric
pip install -r requirements.txt
```
### ▶️ Usage
```bash
python run.py --model your_model_name --task qa
```

### 📊 Evaluation Results

| Model Name     | Accuracy | F1 Score | ROUGE | Notes               |
|----------------|----------|----------|-------|---------------------|
| GPT-4          | 87.5%    | 84.2%    | 0.61  | Zero-shot           |
| LLaMA2-13B     | 75.4%    | 71.8%    | 0.52  | Fine-tuned          |
| Claude 3 Opus  | 79.2%    | 76.5%    | 0.58  | Few-shot setup      |
| Falcon-40B     | 70.1%    | 68.0%    | 0.47  | Baseline            |
| YourModel      | XX.X%    | XX.X%    | XX.X  | Custom results here |
📂 Source: results/cybermetric/scores.csv

