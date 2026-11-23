# Chinese Multimodal News Forgery Benchmark

This repository accompanies the manuscript **A Chinese Multimodal News Dataset with Multiple Forgery Types: Construction and Benchmarking** and provides benchmark code and documentation.

[中文说明](README.zh-CN.md) | [Dataset on Hugging Face](https://huggingface.co/datasets/Wangyq112233/chinese-multimodal-news-forgery)

## Dataset overview

The dataset contains 21,249 Chinese news samples, including 10,739 original news samples and 10,510 samples constructed through controlled forgery operations. It is based on source-traceable, complete news reports collected from the Sina News platform and covers text manipulation, image manipulation, cross-event image–text mismatch, and joint text–image forgery.

Samples undergo automated checks and manual review. Labels describe text modification status, overall forgery status, and specific forgery type according to the construction operations performed. Original news denotes reports that were not modified in this study; it does not imply independent fact-checking of those reports.

## Dataset access

Dataset files are hosted separately on **Hugging Face**:

**[Chinese Multimodal News Forgery Dataset](https://huggingface.co/datasets/Wangyq112233/chinese-multimodal-news-forgery)**

Submit an access request on the dataset page, providing your institution or affiliation and intended use. Requests are reviewed manually. See the dataset card for access conditions, available files, and usage terms.

News texts, images, and modified samples are not distributed through this GitHub repository. Download the required files from Hugging Face after approval and arrange them as described in the benchmark code's README.

## Benchmark tasks

The benchmark defines three tasks:

- Text manipulation detection
- Multimodal forgery detection
- Fine-grained classification

The benchmark subset contains 5,574 samples, divided chronologically by the publication time of the original reports into 3,474 training samples, 1,050 validation samples, and 1,050 test samples.

## Code and data terms

Code licensing and third-party implementation terms are documented in the accompanying `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `LICENSES/` files. The code license does not apply to dataset content.

Dataset access and use are governed by the conditions stated on the Hugging Face dataset page. Third-party news texts and images remain subject to the rights and terms of their respective owners. Access approval does not transfer ownership of, or grant redistribution rights to, those materials or their modified derivatives.

## Citation

Please cite the accompanying manuscript when using the dataset or benchmark:

*A Chinese Multimodal News Dataset with Multiple Forgery Types: Construction and Benchmarking*.

Full bibliographic information will be added when available.
