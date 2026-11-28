# Chinese Multimodal News Forgery Benchmark

This repository accompanies the manuscript **A Chinese Multimodal News Dataset with Multiple Forgery Types: Construction and Benchmarking** and provides benchmark code and documentation.

[中文说明](README.zh-CN.md) | [Benchmark dataset](https://huggingface.co/datasets/Wangyq112233/chinese-multimodal-news-forgery) | [Full dataset](https://huggingface.co/datasets/Wangyq112233/chinese-multimodal-news-forgery-full)

## Dataset overview

The dataset contains 21,249 Chinese news samples, including 10,739 original news samples and 10,510 samples constructed through controlled forgery operations. It is based on source-traceable, complete news reports collected from the Sina News platform and covers text manipulation, image manipulation, cross-event image–text mismatch, and joint text–image forgery.

Samples undergo automated checks and manual review. Labels describe text modification status, overall forgery status, and specific forgery type according to the construction operations performed.

## Dataset access

Dataset files are hosted separately on **Hugging Face**:

- **[Full dataset (21,249 samples)](https://huggingface.co/datasets/Wangyq112233/chinese-multimodal-news-forgery-full)**
- **[Benchmark splits and additional all-image test set](https://huggingface.co/datasets/Wangyq112233/chinese-multimodal-news-forgery)**

Submit an access request on the dataset page, providing your institution or affiliation and intended use. Requests are reviewed manually.

After approval, download the required files from Hugging Face and arrange them as described in the benchmark code's README.

## Benchmark tasks

The benchmark defines three tasks:

- Text manipulation detection
- Multimodal forgery detection
- Fine-grained classification

The benchmark subset contains 5,574 samples, divided chronologically by the publication time of the original reports into 3,474 training samples, 1,050 validation samples, and 1,050 test samples.

## License

Code licensing and third-party implementation terms are documented in `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `LICENSES/`.

Dataset usage conditions are provided on the Hugging Face dataset page.

## Citation

Please cite the accompanying manuscript when using the dataset or benchmark:

*A Chinese Multimodal News Dataset with Multiple Forgery Types: Construction and Benchmarking*.

Full bibliographic information will be added when available.
