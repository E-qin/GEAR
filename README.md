# GEAR: Incorporating Judgment Prediction into Legal Case Retrieval via Law-aware Generative Retrieval

---

![GitHub](https://img.shields.io/github/license/myx666/LeCaRD) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/numpy)

## Overview

* [Background](#background)
* [Getting Started](#getting-started)
* [Dataset Structure](#dataset-structure)

## Background

The implementation of GEAR (**G**enerative L**e**gal C**a**se **R**etrieval).

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
