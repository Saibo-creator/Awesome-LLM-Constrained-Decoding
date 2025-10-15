# Awesome-LLM-Constrained-Decoding

> Towards reliable, controllable and more efficient generation with Large Language Models (LLMs)

A curated list of papers related to constrained decoding of LLM, along with their relevant code and resources.


## Table of Contents

- [Awesome-LLM-Constrained-Decoding](#awesome-llm-constrained-decoding)
  - [Table of Contents](#table-of-contents)
  - [Libraries](#libraries)
  - [Papers](#papers)
  - [Benchmark \& Datasets \& Evaluation](#benchmark--datasets--evaluation)
  - [Survey](#survey)
  - [Blog Posts](#blog-posts)
  - [Related Awesome Lists](#related-awesome-lists)
  - [Disclaimer](#disclaimer)
  - [Contributing](#contributing)

## Libraries


| Library                                                                     | Feature                                                                                | Stars                                                                    |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [guidance-ai/guidance](https://github.com/guidance-ai/guidance)             | CFG, Regex, JSON Schema, Token Forcing, compatible with Transformers, LLAMA-CPP        | ![Stars](https://img.shields.io/github/stars/guidance-ai/guidance)       |
| [outlines-dev/outlines](https://github.com/outlines-dev/outlines)           | CFG, Unicode support, Hugging Face ecosystem, VLLM support                             | ![Stars](https://img.shields.io/github/stars/outlines-dev/outlines)      |
| [eth-sri/lmql](https://github.com/eth-sri/lmql)                             | Regex support, various constraints, more powerful control flow                         | ![Stars](https://img.shields.io/github/stars/eth-sri/lmql)               |
| [jxnl/instructor](https://github.com/jxnl/instructor)                       | Try-Reject-Repeat approach to ensure constraints are met                               | ![Stars](https://img.shields.io/github/stars/jxnl/instructor)            |
| [microsoft/aici](https://github.com/microsoft/aici)                         | A general framework of LLM controller with native support for CFG, Regex, JSON Schema  | ![Stars](https://img.shields.io/github/stars/microsoft/aici)             |
| [noamgat/lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) | Regex, JSON Schema, Beam Search etc.                                                   | ![Stars](https://img.shields.io/github/stars/noamgat/lm-format-enforcer) |
| [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)                       | CFG, careful system optimizations                                                      | ![Stars](https://img.shields.io/github/stars/mlc-ai/xgrammar)            |
| [epfl-dlab/transformers-CFG](https://github.com/epfl-dlab/transformers-CFG) | CFG (EBNF Interface), Compatible with Transformers, Easy to extend for research        | ![Stars](https://img.shields.io/github/stars/epfl-dlab/transformers-CFG) |
| [uiuc-focal-lab/syncode](https://github.com/uiuc-focal-lab/syncode)         | CFG generation that supports builtin grammars like JSON, Python, Go, and more          | ![Stars](https://img.shields.io/github/stars/uiuc-focal-lab/syncode)     |
| [Dan-wanna-M/formatron](https://github.com/Dan-wanna-M/formatron)         | Regex, JSON Schema, CFG, etc           | ![Stars](https://img.shields.io/github/stars/Dan-wanna-M/formatron)     |
| [genlm/genlm-control](https://github.com/genlm/genlm-control) | Arbitrary programmable syntactic and semantic constraints, Constrained decoding as posterior inference, Sequential Monte Carlo | ![Stars](https://img.shields.io/github/stars/genlm/genlm-control)     |
| [structuredllm/itergen](https://github.com/structuredllm/itergen)         |  CFG generation and backtracking to handle semantic constraints          | ![Stars](https://img.shields.io/github/stars/structuredllm/itergen)     |
| [eth-sri/type-constrained-code-generation](https://github.com/eth-sri/type-constrained-code-generation) | TypeScript, including type-safety. | ![Stars](https://img.shields.io/github/stars/eth-sri/type-constrained-code-generation)     |
| [snowkylin/circuit-transformer](https://github.com/snowkylin/circuit-transformer) | Generating logical circuits that are strictly equivalent to given Boolean functions. | ![Stars](https://img.shields.io/github/stars/snowkylin/circuit-transformer)     |
| [epfl-dlab/jsonschemabench](https://github.com/epfl-dlab/jsonschemabench) | A benchmarking framework for evaluating constrained decoding engines on JSON Schema. Supports Guidance, Outlines, XGrammar, OpenAI and more. | ![Stars](https://img.shields.io/github/stars/epfl-dlab/jsonschemabench)     |
| [eth-sri/constrained-diffusion](https://github.com/eth-sri/constrained-diffusion) | Diffusion LLMs, Multi-Region Infilling (CFG) | ![Stars](https://img.shields.io/github/stars/eth-sri/constrained-diffusion) |
| [grammarllm](https://github.com/misaelmongiovi/grammarllm) | LLprefix, DPDA | ![Stars](https://img.shields.io/github/stars/misaelmongiovi/grammarllm)     |


Disclaimer:

- The libraries listed above are **not** exhaustive and are subject to change.
- The features mentioned are **100% not** exhaustive and I strongly recommend checking the respective repositories for more details.
- The libraries are listed by the Github stars
- If you are the author of a library and would like to add or update the information, please open an issue or submit a pull request.

## Papers

Papers with  are newly added papers (not necessarily newly published papers).

**Constrained Decoding for Autoregressive Models**

|  Date   | Paper                                                                                                                                                                                                        |  Publication   |
| :-----: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------: |
| 2025-07 | [Grammar-Constrained Decoding Makes Large Language Models Better Logical Parsers](https://aclanthology.org/2025.acl-industry.34/) | ACL |
| 2025-07 | [GRAMMAR-LLM: Grammar-Constrained Natural Language Generation](https://aclanthology.org/2025.findings-acl.177/) | ACL Findings |
| 2025-07 | [Pre3: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation](https://aclanthology.org/2025.acl-long.551/) | ACL |
| 2025-07 | [Syntactic Control of Language Models by Posterior Inference](https://aclanthology.org/2025.findings-acl.1300/) | ACL Findings |
| 2025-06 | [Earley-Driven Dynamic Pruning for Efficient Structured Decoding](https://arxiv.org/abs/2506.01151) | ICML |
| 2025-05 | [Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling](https://arxiv.org/abs/2504.05410) |  Preprint |
| 2025-05 | [Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo](https://arxiv.org/abs/2504.13139) |  ICLR |
| 2025-04 | [Efficient and Asymptotically Unbiased Constrained Decoding for Large Language Models](https://arxiv.org/abs/2504.09135) |   AISTATS           |
| 2025-02 | [CRANE: Reasoning with constrained LLM generation](https://arxiv.org/abs/2502.09061)                                                                                                        |    ICML      |
| 2025-05 | [Type-Constrained Code Generation with Language Models](https://pldi25.sigplan.org/details/pldi-2025-papers/25/Type-Constrained-Code-Generation-with-Language-Models)                                                                                                       |    PLDI      |
| 2025-02 | [Lost in Space: Optimizing Tokens for Grammar-Constrained Decoding](https://arxiv.org/abs/2502.14969)                                                                                                        |    Preprint      |
| 2025-02 | [Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherence](https://arxiv.org/pdf/2502.14905)                                                                                            |     Preprint      |
| 2025-02 | [Flexible and Efficient Grammar-Constrained Decoding](https://arxiv.org/abs/2502.05111)                                                                                                                      |     Preprint      | 
| 2025-01 | [Circuit Transformer: A Transformer That Preserves Logical Equivalence](https://openreview.net/forum?id=kpnW12Lm9p)                                                                                                |     ICLR      |
| 2025-01 | [Generating Structured Outputs from Language Models: Benchmark and Studies](https://arxiv.org/abs/2501.10868)                                                                                                |     Preprint      |     
| 2024-11 | [XGRAMMAR: FLEXIBLE AND EFFICIENT STRUCTURED GENERATION ENGINE FOR LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2411.15100)                                                                                  |     Preprint      |                
| 2024-10 | [IterGen: Iterative Semantic-aware Structured LLM Generation with Backtracking](https://arxiv.org/abs/2410.07295)                                                                                                                             |     ICLR      |
| 2024-08 | [Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models](https://arxiv.org/html/2408.02442v1)                                                             |     Preprint      |
| 2024-08 | [FANTAstic SEquences and Where to Find Them: Faithful and Efficient API Call Generation through State-tracked Constrained Decoding and Reranking](https://arxiv.org/abs/2407.13945)                          |     Preprint      |
| 2024-07 | [Automata-based constraints for language model decoding](https://arxiv.org/abs/2407.08103)                                                                                                                   |      CoLM      |
| 2024-06 | [Sketch-Guided Constrained Decoding for Boosting Blackbox Large Language Models without Logit Access](https://arxiv.org/abs/2401.09967)                                                                      |      ACL       |
| 2024-05 | [Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047)                                                                                                                                                 |    Preprint    |
| 2024-03 | [SynCode: LLM Generation with Grammar Augmentation](https://arxiv.org/abs/2403.01632)                                                                                                                        |     Preprint      |
| 2024-03 | [Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation](https://icml.cc/virtual/2024/poster/33037)                                                                                           |      ICML      |
| 2024-02 | [Constrained Decoding for Cross-lingual Label Projection](https://arxiv.org/abs/2402.03131)                                                                                                                  |      ICLR      |
| 2024-02 | [Constrained Decoding for Code Language Models via Efficient Left and Right Quotienting of Context-Sensitive Grammars](https://arxiv.org/abs/2402.17988)                                                     |     Preprint      |
| 2024-02 | [Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents](https://arxiv.org/abs/2402.00798)                                                                           |     Preprint      |
| 2023-12 | [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)                                                                                                        |    Preprint    |
| 2023-12 | [Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context](https://papers.nips.cc/paper_files/paper/2023/hash/662b1774ba8845fc1fa3d1fc0177ceeb-Abstract-Conference.html)               |    NeurIPS     |
| 2023-11 | [Prompt Sketching for Large Language Models](https://arxiv.org/abs/2311.04954)                                                                                                                               |    Preprint    |
| 2023-11 | [Sequential Monte Carlo Steering of Large Language Models using Probabilistic Programs](https://arxiv.org/abs/2306.03081)                                                                                    |      PADL      |
| 2023-10 | [Don't Fine-Tune, Decode: Syntax Error-Free Tool Use via Constrained Decoding](https://arxiv.org/abs/2310.07075)                                                                                             |     Preprint      |
| 2023-10 | [Amortizing intractable inference in large language models](https://arxiv.org/abs/2310.04363)                                                                                                                |      ICLR      |
| 2023-10 | [Terminology-Aware Translation with Constrained Decoding and Large Language Model Prompting](https://aclanthology.org/2023.wmt-1.80/)                                                                        |    EMNLP       |
| 2023-10 | [KCTS: Knowledge-Constrained Tree Search Decoding with Token-Level Hallucination Detection](https://aclanthology.org/2023.emnlp-main.867/)                                                                   |     EMNLP      |
| 2023-10 | [Lazy-k Decoding: Constrained Decoding for Information Extraction](https://aclanthology.org/2023.emnlp-main.416/)                                                                                            | EMNLP          |
| 2023-07 | [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702)                                                                                                                    |     Preprint      |
| 2023-06 | [Grammar Prompting for Domain-Specific Language Generation with Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cd40d0d65bfebb894ccc9ea822b47fa8-Abstract-Conference.html) |    NeurIPS     |
| 2023-06 | [Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning](https://aclanthology.org/2023.emnlp-main.674/)                                                                                    |     EMNLP      |
| 2023-06 | [Prompting Is Programming: A Query Language for Large Language Models](https://dl.acm.org/doi/10.1145/3591300)                                                                                               |      PLDI      |
| 2023-05 | [Measuring and Mitigating Constraint Violations of In-Context Learning for Utterance-to-API Semantic Parsing](https://aclanthology.org/2023.findings-emnlp.478/)                                             | EMNLP Findings |
| 2023-04 | [Tractable Control for Autoregressive Language Generation](https://arxiv.org/abs/2304.07438)                                                                                                                 |      ICML      |
| 2022-11 | [Validating Large Language Models with ReLM](https://proceedings.mlsys.org/paper_files/paper/2023/file/93c7d9da61ccb2a60ac047e92787c3ef-Paper-mlsys2023.pdf)                                                 |     MLSys      |
| 2022-11 | [CodePAD: Sequence-based Code Generation with Pushdown Automaton](https://arxiv.org/abs/2211.00818)                                                                                                          |     ISSTA      |
| 2022-05 | [Controllable Text Generation with Neurally-Decomposed Oracle](https://arxiv.org/abs/2205.14219)                                                                                                             |     NeurIPS    |
| 2022-05 | [Gradient-Based Constrained Sampling from Language Models](https://aclanthology.org/2022.emnlp-main.144/)                                                                                                    |     EMNLP      |
| 2022-02 |[COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)                                                                                            |     NeurIPS    |
| 2022-01 | [Synchromesh: Reliable code generation from pre-trained language models](https://arxiv.org/abs/2201.11227)                                                                                                   |      ICLR      |
| 2021-12 | [PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models](https://aclanthology.org/2021.emnlp-main.779/)                                                                 |     EMNLP      |
| 2021-12 | [Constrained Language Models Yield Few-Shot Semantic Parsers](https://aclanthology.org/2021.emnlp-main.608/)                                                                                                 |     EMNLP      |
| 2021-12 | [Controlled Text Generation as Continuous Optimization with Multiple Constraints](https://papers.neurips.cc/paper/2021/file/79ec2a4246feb2126ecf43c4a4418002-Paper.pdf)                                      |    NeurIPS     |
| 2021-06 | [NEUROLOGIC DECODING:(Un)supervised Neural Text Generation with Predicate Logic Constraints](https://aclanthology.org/2021.naacl-main.339.pdf)                                                               |     NAACL      |
| 2019-05 | [A General-Purpose Algorithm for Constrained Sequential Inference](https://aclanthology.org/K19-1045/)                                                                                                       |     CoNLL      |
| 2019-05 | [Improved Lexically Constrained Decoding for Translation and Monolingual Rewriting](https://aclanthology.org/N19-1090/)                                                                                      |     NAACL      |
| 2018-09 | [CGMH: Constrained Sentence Generation by Metropolis-Hastings Sampling](https://arxiv.org/abs/1811.10996)                                                                                                    |      AAAI      |
| 2018-05 | [Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation](https://aclanthology.org/N18-1119/)                                                                        |     NAACL      |
| 2018-04 | [Incorporating Discriminator in Sentence Generation: a Gibbs Sampling Method](https://ojs.aaai.org/index.php/AAAI/article/view/11990)                                                                        |      AAAI      |
| 2017-12 | [Guided Open Vocabulary Image Captioning with Constrained Beam Search](https://aclanthology.org/D17-1098/)                                                                                                   |     EMNLP      |
| 2017-06 | [Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search](https://aclanthology.org/P17-1141/)                                                                                          |      ACL       |


**Constrained Decoding for Diffusion Models**

|  Date   | Paper                                                                                                                                                                                                        |  Publication   |
| :-----: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------: |
| 2025-08 | [Constrained Decoding of Diffusion LLMs with Context-Free Grammars](https://arxiv.org/abs/2508.10111) | Preprint |
| 2025-05 | [DINGO: Constrained Inference for Diffusion LLMs](https://arxiv.org/abs/2505.23061)                | Preprint |
| 2025-03 | [Constrained Language Generation with Discrete Diffusion Models](https://arxiv.org/abs/2503.09790) |  Preprint        |




## Benchmark & Datasets & Evaluation

|  Date   | Paper                                                                                                                                               |               Publication                |
| :-----: | :-------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------: |
| 2025-08 | [JSON-mode Eval Cleaned/Extended, SMILES Eval and HumanEval MRI C++](https://huggingface.co/collections/eth-sri/constrained-diffusion-tasks-689dc25e7ae9a5b6da001e96) | HF Hub |
| 2025-01 | [JsonSchemaBench: Generating Structured Outputs from Language Models: Benchmark and Studies](https://arxiv.org/abs/2501.10868v1) | Preprint |
| 2024-05 | [COLLIE: Systematic Construction of Constrained Text Generation Tasks](https://arxiv.org/abs/2307.08689)                                            |                   ICLR                   |
| 2024-02 | [JSON-mode Eval dataset](https://huggingface.co/datasets/NousResearch/json-mode-eval?row=0)                                                         |                  HF hub                  |
| 2023-12 | [BenchCLAMP: A Benchmark for Evaluating Language Models on Syntactic and Semantic Parsing](https://arxiv.org/abs/2206.10668)                        | NeurIPS Track on Datasets and Benchmarks |
| 2023-10 | [Evaluating Large Language Models on Controlled Generation Tasks](https://arxiv.org/abs/2310.14542)                                                 |                  Preprint                   |
| 2023-09 | [Struc-Bench: Are Large Language Models Really Good at Generating Complex Structured Data?](https://arxiv.org/abs/2309.08963)                       |                  Preprint                   |
| 2021-10 | [NLV corpus](https://nlvcorpus.github.io/)                                                                                                          |                   CHI                    |
| 2020-12 | [CommonGen: A Constrained Text Generation Challenge for Generative Commonsense Reasoning](https://aclanthology.org/2020.findings-emnlp.165.pdf)     |              EMNLP Findings              |
| 2018-09 | [Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task ](https://arxiv.org/abs/1809.08887) |                  EMNLP                   |


## Survey

|  Date    | Paper                                                                                                                                                                               | Publication |
| :-----: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2024-04       | ["We Need Structured Output": Towards User-centered Constraints on Large Language Model Output](https://arxiv.org/abs/2404.07362)                                                                                                                      |   Preprint|

## Blog Posts

- [The good, the bad, and the ugly of Gemini’s structured outputs](https://dylancastillo.co/posts/gemini-structured-outputs.html) by Dylan Castillo
- [Leveraging Constrained Sampling for Fill-in-the-Middle Code Completion](https://blog.nielstron.de/2024/08/14/leveraging-constrained-sampling-for-fill-in-the-middle-code-completion/) by nielstron
- [Proper Well-Formedness for Finite LLM Sampling](https://blog.nielstron.de/2024/08/13/proper-well-formedness-for-finite-llm-sampling/) by nielstron
- [LLM Decoding with Regex Constraints](https://vivien000.github.io/blog/journal/llm-decoding-with-regex-constraints.html) by Vivien
- [Constrained Decoding is Posterior Inference](https://saibo-creator.github.io/post/2024_07_19_constrained_decoding_is_posterior_inference/) by Saibo-creator
- [Making Structured Generation Faster Than Unstructured](https://blog.dottxt.co/selective-multiplication.html)
- [Coding For Structured Generation with LLMs](https://blog.dottxt.co/coding-for-structured-generation.html)
- [Beating GPT-4 with Open Source](https://blog.dottxt.co/oss-v-gpt4.html)
- [Prompt Efficiency - Using Structured Generation to get 8-shot performance from 1-shot.](https://blog.dottxt.co/prompt-efficiency.html)
- [How fast can grammar-structured generation be?](https://blog.dottxt.co/how-fast-cfg.html)
- [Structured Generation Improves LLM performance: GSM8K Benchmark](https://blog.dottxt.co/performance-gsm8k.html)
- [Coalescence: making LLM inference 5x faster](https://blog.dottxt.co/coalescence.html)
- [Constrained Decoding with Arbitrary Constraints is NP-hard](https://saibo-creator.github.io/post/2024_08_25_constrained_decoding_is_np_complete/)
- [LLMs are bad at returning code in JSON](https://aider.chat/2024/08/14/code-in-json.html)
- [Say what I mean](https://github.com/appier-research/structure-gen/blob/main/updates.md)
- [LLGuidance: Making Structured Outputs Go Brrr](https://guidance-ai.github.io/llguidance/llg-go-brrr)

Many of the blogs are written by [Outlines](https://github.com/outlines-dev) team, many thanks to them for their great work! ❤️

## Related Awesome Lists

- [awesome-llm-json](https://github.com/imaurer/awesome-llm-json)

## Disclaimer

This list is not exhaustive and will be updated regularly. If you have any suggestions or want to add a paper, please feel free to open an issue or submit a pull request. We hope to include all relevant papers in this list.

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue. Please make sure to read the [Contributing Guidelines](CONTRIBUTING.md) before contributing.




