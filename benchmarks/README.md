# AI Model Evaluation Benchmarks

This chapter is a curated collection of benchmark datasets and evaluation tools designed to assess the capabilities of custom AI models, particularly in domains related to cybersecurity.

The collection is intended to support researchers and developers who are evaluating their own models using reliable, task-specific benchmarks.

Currently, this are the benchmarks included:

- [SecEval](https://github.com/XuanwuAI/SecEval)
- [CyberMetric](https://github.com/CyberMetric)

The goal is to consolidate diverse evaluation tasks under a single framework to support rigorous, standardized testing.

## 📊 General Summary Table

| Model       | SecEval   | CyberMetric  | Total Value | 
|-------------|-----------|--------------|-------------|
| model_name  | `XX.X%`   | `XX.X%`      | `XX.X%`     | 



## 🔐 [SecEval](https://github.com/XuanwuAI/SecEval)   

#### 📄 Description

SecEval is a benchmark designed to evaluate large language models (LLMs) on security-related tasks. It includes various real-world scenarios such as phishing email analysis, vulnerability classification, and response generation.

#### ▶️ Usage


```bash
git submodule update --init --recursive  # init submodules
cd benchmarks/seceval/eval
pip install -r requirements.txt
```

```bash
python3 eval.py --dataset_file datasets/questions.json --output_dir outputs --backend ollama --model ollama/qwen2.5:14b
```

---

## 🧠 [CyberMetric](https://github.com/CyberMetric)

#### 📄 Description 
CyberMetric is a benchmark framework that focuses on measuring the performance of AI systems in cybersecurity-specific question answering, knowledge extraction, and contextual understanding. It emphasizes both domain knowledge and reasoning ability.


#### ▶️ Usage

```bash
git submodule update --init --recursive  # init submodules
cd benchmarks/cybermetric
python3 CyberMetric_evaluator.py --model_name ollama/qwen2.5:14b --file_path CyberMetric-2-v1.json
```


