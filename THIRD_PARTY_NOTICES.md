# Third-party notices

The root LICENSE applies the MIT License to original benchmark-specific contributions only. This package also includes adaptations of published methods. No single license is asserted for all included third-party components. Retained notices and applicable upstream terms continue to apply.

| Component | Source | Included adaptation |
|---|---|---|
| TextCNN | Yoon Kim, *Convolutional Neural Networks for Sentence Classification*; https://github.com/yoonkim/CNN_sentence | Character tokenization and randomly initialized embeddings in `src/models.py` |
| BERT | https://github.com/google-research/bert | Local Chinese BERT classification and training profile |
| EANN | https://github.com/yaqingwang/EANN-KDD18 | CNN/VGG fusion and gradient reversal, with train-fitted pseudo-event labels |
| CAFE | https://github.com/cyxanna/CAFE | Ambiguity and alignment/detection modules in `src/adapters/cafe_vendor.py`, shared-data adapter, and alternating training |
| HAMMER | https://github.com/rshaojimmy/MultiModal-DeepFake | Vision/BERT compatibility modules and classification adapter |
| ARG | Hu et al., *Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection*, AAAI 2024, DOI: 10.1609/aaai.v38i20.30214 | Model/layer modules in `src/adapters/arg_vendor/` and a shared-data adapter |

`LICENSES/Apache-2.0.txt` retains the Apache 2.0 license text. Original Google, Hugging Face, and NVIDIA copyright/license notices remain in the BERT-derived compatibility file.

`LICENSES/HAMMER-S-Lab-1.0.txt` retains HAMMER's S-Lab License 1.0, including its non-commercial-use restriction. Redistribution of those components remains subject to that license. Absence of a separate bundled license for another upstream component does not grant additional rights.

News text, images, derived samples and pretrained model files are not included in this code-only package. Applicable rights for separately obtained data and model assets remain unchanged.
