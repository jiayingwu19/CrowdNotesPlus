# Data and Code for "Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation"

This repository provides data and code for **CrowdNotes+** — a unified pipeline for **LLM-augmented governance of health misinformation** via Community Notes.  
It includes a preview of our **HealthNotes** dataset and example scripts for **Utility-Guided Note Automation** and evaluation.

---

## Dataset: HealthNotes

A **data preview** is available under `HealthNotes/`, containing:
- 100 samples from the **Helpful** subset
- 100 samples from the **Not Helpful** subset

> The **full dataset** will be released under **gated access** upon acceptance of the paper (see the Ethical Considerations section of our manuscript).

---

## Note Automation & Evaluation

- The main automation pipeline is implemented in `main.py`.
- Example evaluation outputs and generated notes are provided in `automation_examples.tar.gz`, covering 100 Helpful and 100 Not Helpful samples.
- This setting follows the **Utility-Guided Note Automation** mode described in the paper, where models generate candidate notes that are automatically evaluated for **evidence relevance**, **correctness**, and **helpfulness**.

---

## Models Evaluated

We evaluate a set of fifteen representative LLMs across closed-source, open-source, and domain-specific categories (see Section 6 of the manuscript for full results).

| **Model**          | **Model Card**                               |
|:-------------------|:---------------------------------------------|
| Gemini-2.5-Pro     | `gemini-2.5-pro-preview-03-25`               |
| o3                 | `o3-2025-04-16`                              |
| Grok-4             | `x-ai/grok-4`                                |
| GPT-4.1            | `gpt-4.1-2025-04-14`                         |
| Claude-Opus-4      | `claude-opus-4-20250514`                     |
| Qwen3-32B          | `Qwen/Qwen3-32B`                             |
| Qwen3-14B          | `Qwen/Qwen3-14B`                             |
| Llama-3.1-8B       | `meta-llama/Llama-3.1-8B-Instruct`           |
| Ministral-8B       | `mistralai/Ministral-8B-Instruct-2410`       |
| Qwen3-8B           | `Qwen/Qwen3-8B`                              |
| Lingshu-32B        | `lingshu-medical-mllm/Lingshu-32B`           |
| MedGemma-27B       | `google/medgemma-27b-text-it`                |
| Lingshu-7B         | `lingshu-medical-mllm/Lingshu-7B`            |
| MedGemma-4B        | `google/medgemma-4b-it`                      |



