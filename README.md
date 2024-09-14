# [SIGIR '24]Explicitly Integrating Judgment Prediction with Legal Document Retrieval: A Law-Guided Generative Approach

---

![GitHub](https://img.shields.io/github/license/myx666/LeCaRD) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/numpy)

## Overview

* [Background](#background)

- [Getting Started](#getting-started)

* [Dataset Structure](#dataset-structure)

## Background

The implementation of SIGIR'24 papar ([Explicitly Integrating Judgment Prediction with Legal Document Retrieval: A Law-Guided Generative Approach](https://dl.acm.org/doi/10.1145/3626772.3657717)).

## Getting Started

### 1. Requirements

Please refer to `/GEAR/env.yaml`

### 2. Commands

As shown in  `/GEAR/ex/ex_history.sh`, the commands to run are:

```shell
# ELAM dataset
nohup accelerate launch train.py -c ex/config/ELAM/GO.json  > ex/log/ELAM/GO.log 2>&1 &  # bf16

# LeCaRDv2 dataset
nohup accelerate launch train_LeCaRDv2_join.py -c ex/config/LeCaRD_version2/GO.json > ex/log/LeCaRD_version2/GO.log 2>&1 &  # bf16
```

## Dataset Structure

`/GEAR/dataset` is the root directory of the two datasets on which the experiments are based.`/GEAR/law` is the directory of legal-related documents. The meanings of some directories or files are introduced below:

```python
GEAR
├── dataset
│   ├── ELAM   
│   │   ├── ELAM_key_src                # Key source folder of ELAM including text, labels, 
│   │   │                               #     charge information, and other relevant data
│   │   │				#     sources for query and precedent cases.
│   │   └── QGen                        # Sentence-level score files for rationale extraction.
│   └── LeCaRD_version2
│       ├── LeCaRD_version2_key_src     # ...
│       └── QGen                        # ...
└── law
    ├── Criminal Law.txt                # Criminal Law of the People's Republic of China.
    ...
```

## Ref

If you find our work useful, please do not save your star and cite our work:

```
@inproceedings{Qin2024Explicitly,
author = {Qin, Weicong and Cao, Zelin and Yu, Weijie and Si, Zihua and Chen, Sirui and Xu, Jun},
title = {Explicitly Integrating Judgment Prediction with Legal Document Retrieval: A Law-Guided Generative Approach},
year = {2024},
isbn = {9798400704314},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3626772.3657717},
doi = {10.1145/3626772.3657717},
booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2210–2220},
numpages = {11},
keywords = {generative retrieval, legal document retrieval, legal judgment prediction},
location = {Washington DC, USA},
series = {SIGIR '24}
}
```