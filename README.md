# Chinese Multimodal News Forgery Dataset

This repository accompanies the manuscript **A Chinese Multimodal News Dataset with Multiple Forgery Types: Construction and Benchmarking**.

The study presents a Chinese news dataset constructed from source-traceable reports collected from the Sina News platform. It includes original news reports and four categories of controlled forgery: text manipulation, image manipulation, cross-event image–text mismatch, and joint text–image forgery. The labels describe the construction operations performed in this study.

## Current Availability

This repository currently contains project documentation and a data-format illustration only. Dataset files and benchmark evaluation code have not yet been uploaded and are not currently available for download. The repository does not currently provide the materials needed to reproduce the reported benchmark results.

## Dataset Format

The dataset uses JSON records. The intended streamlined data format contains the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | Integer | Unique identifier for each news sample. |
| `title` | String | News headline. |
| `content` | String | News body text. |
| `images` | Object | Dictionary containing relative paths to associated image files. |
| `url` | String | Source URL of the original news report. |

The `images` field may contain multiple entries, such as `image1`, `image2`, and `image3`. An empty object (`{}`) indicates that no image is available for the corresponding sample.

The accompanying file **`sample_structure.json`** illustrates the intended record structure using null placeholders. It does not contain actual news text, image paths, source URLs, or experimental records.

The study uses fixed training, validation, and test sets. The corresponding split assignments and task-specific labels are planned for release separately. The structure example alone is not sufficient to reproduce the benchmark experiments.

## Label Definitions

The study defines five categories according to the operations performed during dataset construction.

| Forgery Code | Category | Description |
|---|---|---|
| `0` | Original News | News reports retained without controlled modifications. |
| `110` | Text Manipulation | Modification of the news text while retaining the original images, if available. |
| `101` | Image Manipulation | Modification of visual content while preserving the original news text. |
| `102` | Cross-Event Image–Text Mismatch | Replacement of an image with an unedited image from another news event. |
| `111` | Joint Text–Image Forgery | Modification of both textual and visual content. |

Three label fields are defined for the benchmark tasks:

- `forgery`: Fine-grained category label identifying the construction operation.
- `mlabel`: Binary label indicating whether a controlled forgery operation has been performed.
- `label`: Binary label indicating whether the news text has been modified.

These labels are defined according to the dataset construction procedures rather than independent verification of the factual accuracy of the original news reports.

The labels and fixed data splits are not currently distributed through this repository.

## Planned Release

We plan to make the following materials publicly available upon publication of the paper:

- Dataset labels and sample identifiers
- The fixed training, validation, and test splits used in the paper
- Source indices and references to the original reports, where sharing is permitted
- Documentation of the construction procedures and label definitions
- Benchmark evaluation code and configuration files

This list describes a planned release, not materials that are already available. This README will be updated with file descriptions and access instructions as materials are released.

## News Texts and Images

The source news texts and images originate from third parties. Their inclusion in the study does not imply that we hold the right to redistribute them. Public access to an original webpage does not, by itself, establish permission to redistribute its contents.

The availability of full news texts, images, and their modified derivatives will depend on the applicable third-party permissions and licensing terms. These files are not currently distributed through this repository.

If a controlled-access arrangement can be established, its scope, eligibility conditions, and application procedure will be documented here. No such access process is currently offered through this repository.

## Licensing

No dataset or code license is granted by this release notice. Licensing information for materials released in the future will be specified alongside those materials. Third-party content will remain subject to the rights and terms of its respective owners.

## Repository

https://github.com/tiepiw/chinese-multimodal-news-forgery

## Citation

If you find this dataset or repository useful for your research, please cite our paper once it becomes available.

**Paper Title:** *A Chinese Multimodal News Dataset with Multiple Forgery Types: Construction and Benchmarking*

The formal citation information, including publication venue and DOI, will be updated once the paper is published.

## Contact

For questions regarding the dataset, research collaboration, or academic inquiries, please contact:

**Email:** wyq6696@gsupl.edu.cn

**Affiliation:** School of Artificial Intelligence, Gansu University of Political Science and Law, China.
