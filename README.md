# SemEval-2026 Task 13: Detecting Machine-Generated Code with Multiple Programming Languages, Generators, and Application Scenarios

## 🔍 Task Overview

The rise of generative models has made it increasingly difficult to distinguish machine-generated code from human-written code — especially across different programming languages, domains, and generation techniques. 

**SemEval-2026 Task 13** challenges participants to build systems that can **detect machine-generated code** under diverse conditions by evaluating generalization to unseen languages, generator families, and code application scenarios.

The task consists of **three subtasks**:

---

### Subtask A: Binary Machine-Generated Code Detection

**Goal:**  
Given a code snippet, predict whether it is:

- **(i)** Fully **human-written**, or  
- **(ii)** Fully **machine-generated**

**Training Languages:** `C++`, `Python`, `Java`  
**Training Domain:** `Algorithmic` (e.g., Leetcode-style problems)

**Evaluation Settings:**

| Setting                              | Language                | Domain                 |
|--------------------------------------|-------------------------|------------------------|
| (i) Seen Languages & Seen Domains    | C++, Python, Java       | Algorithmic            |
| (ii) Unseen Languages & Seen Domains | Go, PHP, C#, C, JS      | Algorithmic            |
| (iii) Seen Languages & Unseen Domains| C++, Python, Java       | Research, Production   |
| (iv) Unseen Languages & Domains      | Go, PHP, C#, C, JS      | Research, Production   |

**Dataset Size**: 
- Train - 500K samples (238K Human-Written | 262K Machine-Generated)
- Validation - 100K samples
---

###  Subtask B: Multi-Class Authorship Detection

**Goal:**  
Given a code snippet, predict its author:

- **(i)** Human  
- **(ii–xi)** One of 10 LLM families:
  - `DeepSeek-AI`, `Qwen`, `01-ai`, `BigCode`, `Gemma`, `Phi`, `Meta-LLaMA`, `IBM-Granite`, `Mistral`, `OpenAI`

**Evaluation Settings:**

- **Seen authors**: Test-time generators appeared in training  
- **Unseen authors**: Test-time generators are new but from known model families

**Dataset Size**: 
- Train - 500K samples (442K Human |4K DeepSeek-AI | 8K Qwen| 3K 01-ai |2 K BigCode |2K Gemma | 5K Phi | 8K Meta-LLaMA |8K IBM-Granite| 4K  Mistral   |10K OpenAI)
- Validation - 100K samples
---

### Subtask C: Hybrid Code Detection

**Goal:**  
Classify each code snippet as one of:

1. **Human-written**  
2. **Machine-generated**  
3. **Hybrid** — partially written or completed by LLM  
4. **Adversarial** — generated via adversarial prompts or RLHF to mimic humans

**Dataset Size**: 
- Train - 900K samples (485K Human-written | 210K Machine-generated |  85K Hybrid | 118K Adversarial)
- Validation - 200K samples
---

## 📁 Data Format

- All data will be released via:
  - [Kaggle](https://www.kaggle.com/datasets/daniilor/semeval-2026-task13)  
  - [HuggingFace Datasets](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13)
  - In this GitHub repo as `.parquet` file

- For each subtask:
  - Dataset contains `code`,  `label` (which is label id), and additional meta-data such as programming language (`language`), and the `generator`.
  - Label mappings (`label_to_id.json` and `id_to_label.json`) are provided in each task folder  

---
## 🔒 Data and Model Restrictions

- The use of **additional training data is not allowed**. Participants must use only the official training sets provided for each subtask.
- It is also **not permitted to use models that have been pre-trained specifically for AI-generated code detection** by third parties.
- However, participants are allowed to use **general-purpose or code-oriented pre-trained models** (e.g., CodeBERT, StarCoder, etc.)

Please adhere strictly to these rules to ensure a fair comparison across submissions. If you have any doubts, contact task organizers

---

## 📤 Submission Format

- Submit a `.csv` file with two columns:
  - `id`: unique identifier of the code snippet  
  - `label`: the **label ID** (not the string label)

- Sample submission files are available in each task’s folder  
- A **single scorer script** (`scorer.py`) is used for all subtasks  
- Evaluation measure: **macro F1** for all subtasks

## 📢 Kaggle competition
The Kaggle competitions for the SemEval task are now live! 
You can submit your system outputs using the following links:

* [Task A](https://www.kaggle.com/t/99673e23fe8546cf9a07a40f36f2cc7e)

* [Task B](https://www.kaggle.com/t/65af9e22be6d43d884cfd6e41cad3ee4)

* [Task C](https://www.kaggle.com/t/005ab8234f27424aa096b7c00a073722)

At the moment, only the **public test set** is available. The leaderboard shown now is for convenience only - it reflects results on the **public test set**.
We will release the **private test set on Jan. 10**, which will be used for the **final evaluation and ranking**.

Please make sure to **resubmit** your final predictions once the private test set is released, as only those submissions will be considered for the official evaluation. We will inform all participants when the private test data becomes available.


## FAQs
> **Q1: What’s the participation process and how do I register?**

We will release our Kaggle website soon for participant registration. You can register anytime before the evaluation phase begins. Once registered, simply prepare your detection results and submit them before the evaluation deadline.

> **Q2: There are three tasks. Do I have to participate in all of them, or can I choose?**

You are free to participate in one, two, or all three tasks—it’s completely up to you.

> **Q3: What are the important dates for the project?**

We aim to align all key dates with the official SemEval committee schedule and will announce them accordingly.

> **Q4: What methods and technologies can I use?**

Be creative! The only restriction is that we **do not allow** the usage of AI-generated content detectors, trained by third parties and restrict the data usage to the provided training sets. Other than that, feel free to use any methods and technologies you prefer.

>**Q5: How do I start?**

You can use [Starter Files](https://github.com/mbzuai-nlp/SemEval-2026-Task13/tree/main/baselines/Kaggle_starters) to have some direction of work. You may experiment with backbone models, training strategy etc. Also feel free to ask questions and share your ideas on **Discussion** page of Kaggle competitions.

## Important Dates
- ~~Sample data ready: 15 July 2025~~
- ~~Training data ready: **1 September 2025**~~
- **Evaluation data ready: 1 December 2025** (we already released the training and validation datasets) 
- Evaluation data ready and evaluation start: 10 January 2026 (we will share private test data at this time)
- Evaluation end: 24 January 2026
- Paper submission due: February 2026
- Notification to authors: March 2026
- Camera ready due April 2026
- SemEval workshop Summer 2026 (co-located with a major NLP conference)


## Citation
Our task is based on enriched data from our previous works. Please, consider citing them, when using data from this task

Droid: A Resource Suite for AI-Generated Code Detection
```
@misc{orel2025textttdroidresourcesuiteaigenerated,
      title={$\texttt{Droid}$: A Resource Suite for AI-Generated Code Detection}, 
      author={Daniil Orel and Indraneil Paul and Iryna Gurevych and Preslav Nakov},
      year={2025},
      eprint={2507.10583},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2507.10583}, 
}
```

CoDet-M4: Detecting Machine-Generated Code in Multi-Lingual, Multi-Generator and Multi-Domain Settings
```
@inproceedings{orel-etal-2025-codet,
    title = "{C}o{D}et-M4: Detecting Machine-Generated Code in Multi-Lingual, Multi-Generator and Multi-Domain Settings",
    author = "Orel, Daniil  and
      Azizov, Dilshod  and
      Nakov, Preslav",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.550/",
    pages = "10570--10593",
    ISBN = "979-8-89176-256-5",
}
```

