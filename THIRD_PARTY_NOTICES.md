# Third-party sources and notices

This file records method references, implementation sources, and adaptations used in this benchmark. Original benchmark-specific code is covered by the root MIT License; third-party code retains its upstream license and source-file notices.

## Method references

| Method | Reference | Benchmark implementation |
|---|---|---|
| TextCNN | Yoon Kim, *Convolutional Neural Networks for Sentence Classification*; [author repository](https://github.com/yoonkim/CNN_sentence) | Character-level CNN with randomly initialized embeddings in `src/models.py` |
| EANN | [EANN-KDD18](https://github.com/yaqingwang/EANN-KDD18) | CNN/VGG fusion, gradient reversal, and train-fitted pseudo-event supervision in `src/models.py` |

These links identify the methods used for the benchmark implementations.

## Implementation sources and adaptations

| Component | Source | Included implementation or adaptation |
|---|---|---|
| BERT | [Hugging Face Transformers](https://github.com/huggingface/transformers); [chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext) | Chinese BERT classification using the external Transformers package and locally supplied pretrained weights |
| CAFE | [cyxanna/CAFE](https://github.com/cyxanna/CAFE) | Modules in `src/adapters/cafe_vendor.py` and the shared-data adapter in `src/adapters/cafe.py` |
| HAMMER | [rshaojimmy/MultiModal-DeepFake](https://github.com/rshaojimmy/MultiModal-DeepFake) | Vision and BERT modules in `src/adapters/hammer_vendor/` and the classification adapter in `src/adapters/hammer.py` |
| ARG | [ICTMCG/ARG](https://github.com/ICTMCG/ARG), official repository for *Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection*, AAAI 2024 | Model and layer modules in `src/adapters/arg_vendor/` and the shared-data adapter in `src/adapters/arg.py` |

## Retained licenses

- `LICENSES/HAMMER-S-Lab-1.0.txt`: S-Lab License 1.0 for HAMMER-derived components, including its non-commercial-use condition.
- `LICENSES/Apache-2.0.txt`: Apache License 2.0 for components carrying the corresponding notices.
- Google, Hugging Face, and NVIDIA copyright notices are retained in `src/adapters/hammer_vendor/xbert.py`.

## Benchmark-specific components

Concat and HIMF-MoE are benchmark-specific implementations. The Qwen evaluation code calls an external Ollama service; Qwen model weights are obtained separately.
