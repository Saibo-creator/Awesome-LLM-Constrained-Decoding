# Awesome-LLM-Constrained-Decoding

A curated list of papers related to constrained decoding of LLM, along with their relevant code and resources.


## Table of Contents

- [Awesome-LLM-Constrained-Decoding](#awesome-llm-constrained-decoding)
  - [Table of Contents](#table-of-contents)
  - [Papers](#papers)
  - [Benchmark \& Datasets](#benchmark--datasets)
  - [Survey](#survey)
  - [Disclaimer](#disclaimer)
  - [Contributing](#contributing)


## Papers


|  Date    | Paper                                                                                                                                                                               | Publication |
| :-----: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2024-06 | [Sketch-Guided Constrained Decoding for Boosting Blackbox Large Language Models without Logit Access](https://arxiv.org/abs/2401.09967)                                                                                                                    |  ACL <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fd9ef08a1a0d231d01ce805c61230995877c606d3%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2024-05 | [Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047)                                                                                                                   |  Preprint <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb95ca121a606da32180ab8bb0c0c58bf19b1499b%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2024-03 | [Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation](https://icml.cc/virtual/2024/poster/33037)                                                                                                                     |  ICML<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb95ca121a606da32180ab8bb0c0c58bf19b1499b%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2024-2 | [Constrained Decoding for Code Language Models via Efficient Left and Right Quotienting of Context-Sensitive Grammars](https://arxiv.org/abs/2402.17988)                                                                                                                   |  Arxiv <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F2be4efd184deaa10d69d58ab87d6032680e32931%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2023-12 | [Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context](https://papers.nips.cc/paper_files/paper/2023/hash/662b1774ba8845fc1fa3d1fc0177ceeb-Abstract-Conference.html)                                                                                                                  |  NeurIPS <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9b405bcf0c10a220e848eed43573ffc3477e13b8%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2023-10  | [Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning](https://aclanthology.org/2023.emnlp-main.674/)                                                                                                                    |  EMNLP<br> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F7e269bfabb451765a16ca0357de6b497cefb60bf%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2023-10 | [Grammar Prompting for Domain-Specific Language Generation with Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cd40d0d65bfebb894ccc9ea822b47fa8-Abstract-Conference.html)                                                                                                                      |  NeurIPS<br> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Faf705d648b5b16daa3dcc593bc593f2574d76c07%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2023-10 | [Don't Fine-Tune, Decode: Syntax Error-Free Tool Use via Constrained Decoding](https://arxiv.org/abs/2310.07075)                                                                                                                  |  Arxiv <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9c55dd7af13d6a53832e0117fc2c8ec55e386751%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2023-10 | [KCTS: Knowledge-Constrained Tree Search Decoding with Token-Level Hallucination Detection](https://aclanthology.org/2023.emnlp-main.867/)                                                                                                                  |  EMNLP<br> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe6f74f2746a9e8bc90701f2afcf3c47e5e98b2dd%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2023-07 | [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702) | Arxiv <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fc4ceaef35bca063815f50d90a087acbd07a65478%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2023-06       | [Prompting Is Programming: A Query Language for Large Language Models](https://dl.acm.org/doi/10.1145/3591300)                                                                                                                      |   PLDI<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fc2329c685f11efa25c562f97be71ff03103423fd%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2022-11 | [Validating Large Language Models with ReLM](https://proceedings.mlsys.org/paper_files/paper/2023/file/93c7d9da61ccb2a60ac047e92787c3ef-Paper-mlsys2023.pdf)                                                                                                                   |  MLSys<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ffd291f8ab240af366da0133c5c5b31562484d5a3%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2022-11 | [CodePAD: Sequence-based Code Generation with Pushdown Automaton](https://arxiv.org/abs/2211.00818) |  ISSTA <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F3cba16fc46ac5b35c1cc72a822208aa0097384cc%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2022-05| [Gradient-Based Constrained Sampling from Language Models](https://aclanthology.org/2022.emnlp-main.144/)                                                                                                                  |  EMNLP <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ff1e56def812bc398d1b2b8c9a7ea6a623abd38e5%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2022-01 | [Synchromesh: Reliable code generation from pre-trained language models](https://arxiv.org/abs/2201.11227)                                                                                                                   |  ICLR<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb62d63580b81a2cbb20c3c1593dd62d118e4cb07%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2021-12 | [PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models](https://aclanthology.org/2021.emnlp-main.779/) |  EMNLP <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F5fbcfccd3736969d95ed660d8e6962c86b7a9113%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2021-12  | [Constrained Language Models Yield Few-Shot Semantic Parsers](https://aclanthology.org/2021.emnlp-main.608/) |  EMNLP <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F64a1dbdd7653eaca25c78e87335ee156b6f6959e%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2021-12 | [Controlled Text Generation as Continuous Optimization with Multiple Constraints](https://papers.neurips.cc/paper/2021/file/79ec2a4246feb2126ecf43c4a4418002-Paper.pdf) |  NeurIPS <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F3b02eb62f198449840334da26917e3610b283e25%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2021-06 |[NEUROLOGIC DECODING:(Un)supervised Neural Text Generation with Predicate Logic Constraints](https://aclanthology.org/2021.naacl-main.339.pdf) | NAACL <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F2c5bf29079cd958a2bef150077a02a1deb300652%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2019-05 | [A General-Purpose Algorithm for Constrained Sequential Inference](https://aclanthology.org/K19-1045/) |  CoNLL <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fd013ec4049e82ba1002c9ea2c6a0c411890757dd%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2019-05 | [Improved Lexically Constrained Decoding for Translation and Monolingual Rewriting](https://aclanthology.org/N19-1090/) |  NAACL <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F101141b047d119ef9c8fda8dd83d3d9eb3fbfd1f%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2018-05 | [Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation](https://aclanthology.org/N18-1119/) |  NAACL <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F823f335eee85b42502c8c6cb3ce38b4ae274ef89%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2017-12 | [Guided Open Vocabulary Image Captioning with Constrained Beam Search](https://aclanthology.org/D17-1098/) | EMNLP <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F086fa2fe3ee2a5b805aeaf9fbfe59ee8157dad5c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2017-06 | [Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search](https://aclanthology.org/P17-1141/) |  ACL <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F82f9637e263251b2387c8e0c87b942bd1b6c3bdd%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |



## Benchmark & Datasets

|  Date    | Paper                                                                                                                                                                               | Publication |
| :-----: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2024-05 | [COLLIE: Systematic Construction of Constrained Text Generation Tasks](https://arxiv.org/abs/2307.08689)                                                                                                                   |  ICLR <br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F18b75ea107ed166d7120c12c162b94f02e20b417%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2023-12 | [BenchCLAMP: A Benchmark for Evaluating Language Models on Syntactic and Semantic Parsing](https://arxiv.org/abs/2206.10668)                                                                                                                   |  NeurIPS Track on Datasets and Benchmarks <br> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F95e2f656017f9ec5d9cd411b1f744b278131ce6c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2020-12 | [CommonGen: A Constrained Text Generation Challenge for Generative Commonsense Reasoning](https://aclanthology.org/2020.findings-emnlp.165.pdf)                                                                                                                  |  EMNLP Findings <br> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F0119a57cf88ef16e6dc291252fae340bb6b3953c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |



## Survey

|  Date    | Paper                                                                                                                                                                               | Publication |
| :-----: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2024-04       | ["We Need Structured Output": Towards User-centered Constraints on Large Language Model Output](https://arxiv.org/abs/2404.07362)                                                                                                                      |   Arxiv<br> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fa78f0c4433a34b90fe4a62c226b79af68d7d7deb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)  |


## Disclaimer

This list is not exhaustive and will be updated regularly. If you have any suggestions or want to add a paper, please feel free to open an issue or submit a pull request. We hope to include all relevant papers in this list.

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue. Please make sure to read the [Contributing Guidelines](CONTRIBUTING.md) before contributing.




