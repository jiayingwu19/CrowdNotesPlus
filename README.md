# Data and Code for "Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation" (ACL 2026 Oral)

This repo contains the data and code for the following paper:

Jiaying Wu*, Zihang Fu*, Haonan Wang, Fanxiao Li, Jiafeng Guo, Preslav Nakov, Min-Yen Kan. Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation, The 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026). [[Paper PDF]](https://arxiv.org/pdf/2510.11423)

## Abstract

Community Notes, the crowd-sourced misinformation governance system on X (formerly Twitter), allows users to flag misleading posts, attach contextual notes, and rate the notes' helpfulness. However, our empirical analysis of 30.8K health-related notes reveals substantial latency, with a median delay of 17.6 hours before notes receive a helpfulness status. To improve responsiveness during real-world misinformation surges, we propose CrowdNotes+, a unified LLM-based framework that augments Community Notes for faster and more reliable health misinformation governance. CrowdNotes+ integrates two modes: (1) evidence-grounded note augmentation and (2) utility-guided note automation, supported by a hierarchical three-stage evaluation of relevance, correctness, and helpfulness. We instantiate the framework with HealthNotes, a benchmark of 1.2K health notes annotated for helpfulness, and a fine-tuned helpfulness judge. Our analysis first uncovers a key loophole in current crowd-sourced governance: voters frequently conflate stylistic fluency with factual accuracy. Addressing this via our hierarchical evaluation, experiments across 15 representative LLMs demonstrate that CrowdNotes+ significantly outperforms human contributors in note correctness, helpfulness, and evidence utility.

---

## Data: the HealthNotes benchmark

Our HealthNotes benchmark contains 1,268 instances and is available under `HealthNotes/`. Each instance consists of one flagged misleading post, a human-written Community Note with evidence URLs, and its crowd-voted helpfulness status (`Helpful` or `Not Helpful`).

The data are split by helpfulness status: `data/helpful_634_samples.jsonl` contains the 634 samples labeled as `Helpful`, while `data/not_helpful_634_samples.jsonl` contains the 634 samples labeled as `Not Helpful`.

Each data instance follows the format below:

```json
{
  "noteId": "[ID of the Community Note]",
  "tweet": "textual content of the flagged post",
  "note": "human-written Community Note with evidence URLs",
  "createdAtMillis": "note publication timestamp in milliseconds",
  "helpfulnessStatus": "crowd-voted helpfulness status"
}
```

---

## CrowdNotes+: Note Generation and Evaluation

The implementation of our CrowdNotes+ framework for LLM-augmented Community Notes is provided under `crowdnotesplus/`. Please refer to `crowdnotesplus/README.md` for details on the project layout and usage.

We also provide example outputs under the utility-guided note automation mode in `automation_examples.tar.gz`. These examples include evaluation outputs and generated notes for 100 `Helpful` samples and 100 `Not Helpful` samples.

---

## Base LLM Configuration

We evaluate fifteen representative LLMs across closed-source, open-source, and domain-specific categories. Please refer to Section 6 of the manuscript for the full experimental results.

| **Model**      | **Model Card**                         |
| :------------- | :------------------------------------- |
| Gemini-2.5-Pro | `gemini-2.5-pro-preview-03-25`         |
| o3             | `o3-2025-04-16`                        |
| Grok-4         | `x-ai/grok-4`                          |
| GPT-4.1        | `gpt-4.1-2025-04-14`                   |
| Claude-Opus-4  | `claude-opus-4-20250514`               |
| Qwen3-32B      | `Qwen/Qwen3-32B`                       |
| Qwen3-14B      | `Qwen/Qwen3-14B`                       |
| Llama-3.1-8B   | `meta-llama/Llama-3.1-8B-Instruct`     |
| Ministral-8B   | `mistralai/Ministral-8B-Instruct-2410` |
| Qwen3-8B       | `Qwen/Qwen3-8B`                        |
| Qwen3-8B (reasoning-enabled)       | `Qwen/Qwen3-8B`                        |
| Lingshu-32B    | `lingshu-medical-mllm/Lingshu-32B`     |
| MedGemma-27B   | `google/medgemma-27b-text-it`          |
| Lingshu-7B     | `lingshu-medical-mllm/Lingshu-7B`      |
| MedGemma-4B    | `google/medgemma-4b-it`                |

---

## Hierarchical Note Helpfulness Evaluation

We propose a hierarchical note helpfulness evaluation scheme with high human alignment. The evaluation consists of three progressive gates: (1) evidence relevance, (2) evidence representation correctness, and (3) note helpfulness. A note is deemed helpful only if it passes all three gates.

The first two gates are implemented using GPT-4.1-based judges. The corresponding prompts are provided in `crowdnotesplus/prompts.py`.

For the final helpfulness gate, we introduce HealthJudge, a Lingshu-7B model fine-tuned on post-note pairs for helpfulness judgment. HealthJudge is available on [Hugging Face](https://huggingface.co/Eculid/HealthJudge).

## Contact

jiayingwu [at] u.nus.edu

## Citation

If you find this repo useful for your research, please consider citing our paper:

```
@article{wu2025beyond,
  title={Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation},
  author={Wu, Jiaying and Fu, Zihang and Wang, Haonan and Li, Fanxiao and Guo, Jiafeng and Nakov, Preslav and Kan, Min-Yen},
  journal={arXiv preprint arXiv:2510.11423},
  year={2025}
}
```
