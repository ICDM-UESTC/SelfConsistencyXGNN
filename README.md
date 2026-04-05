# Self-Consistency Improves the Trustworthiness of Self-Interpretable GNNs

**💻 Official implementation of our ICLR 2026 paper: Self-Consistency Improves the Trustworthiness of Self-Interpretable GNNs**

> 🧠 Authors: [Wenxin Tai](https://scholar.google.com/citations?user=YyxocAIAAAAJ&hl=en), [Ting Zhong](https://scholar.google.com/citations?user=Mdr0XDkAAAAJ&hl=en), [Goce Trajcevski](https://scholar.google.com/citations?user=Avus2kcAAAAJ&hl=en), [Fan Zhou](https://scholar.google.com/citations?user=Ihj2Rw8AAAAJ&hl=en)  
> 📍 Institutions: University of Electronic Science and Technology of China & Iowa State University  
> 🔗 [Page Link](https://icdm-uestc.github.io/SelfConsistencyXGNN)
> 🔗 [Paper Link](https://iclr.cc/virtual/2026/poster/10008008)
> 🤖 This repository is maintained by [ICDM Lab](https://www.icdmlab.com/)

---

## 🧩 Overview

TL;DR: We identify the mismatch between SI-GNN training and faithfulness evaluation, show its connection to self-consistency, and propose a simple SC loss that consistently improves explanation quality without architectural changes.

---

## 📦 Repository Structure

```bash
├── assets
├── configs         # configuration
├── criterion.py    # loss function
├── dataloader.py   # load data
├── dataset.py      # process data
├── datasets        # raw dataset
├── explainer.py    # explainer in self-interpretable GNNs (MLP)
├── main.py         # entry
├── model.py        # GNN backbone (GIN)
├── outputs         # checkpoints/logs
├── README.md
├── run.sh 
└── trainer.py      # train/valid/test
```

---

## ⚙️ Installation

We recommend creating a fresh Python environment (e.g., with conda):

```bash
conda create -n exgnn python=3.9
conda activate exgnn
pip install -r requirements.txt
```

---

## 📚 Datasets

We evaluate our method on a variety of datasets:

* Synthetic: BA-2MOTIFS
* Molecular: MUTAGENICITY, 3MR, BENZENE

Datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1RaOKbWABerHfea_sJZGIbXSy0FcOzK0O?usp=sharing), place all datasets (e.g., `ba_2motifs`, `benzene`, `mr`, `mutag`) in the `datasets/` folder.  

---

## 🏃‍♀️ Quick Start

### 1. Train self-interpretable GNNs

```bash
python main.py --run_time 10 --dataset ba_2motifs --method size
python main.py --run_time 10 --dataset ba_2motifs --method size_sc
python main.py --run_time 10 --dataset ba_2motifs --method gsat
python main.py --run_time 10 --dataset ba_2motifs --method gsat_sc
python main.py --run_time 10 --dataset ba_2motifs --method cal_cr
python main.py --run_time 10 --dataset ba_2motifs --method cal_cr_sc
```


### 2. Evaluation

```bash
python main.py --run_time 10 --dataset ba_2motifs --method size --calculate_all_metrics
python main.py --run_time 10 --dataset ba_2motifs --method size_sc --calculate_all_metrics
python main.py --run_time 10 --dataset ba_2motifs --method gsat --calculate_all_metrics
python main.py --run_time 10 --dataset ba_2motifs --method gsat_sc --calculate_all_metrics
python main.py --run_time 10 --dataset ba_2motifs --method cal_cr --calculate_all_metrics
python main.py --run_time 10 --dataset ba_2motifs --method cal_cr_sc --calculate_all_metrics
```

---

## 📁 Pretrained Checkpoints
We provide **pretrained model checkpoints** for quick reproduction.

You can download them from the [Releases](https://github.com/ICDM-UESTC/SelfConsistencyXGNN/releases) tab

To use the checkpoint, place it in the `outputs/checkpoints/` folder and run:

```bash
python main.py --run_time 10 --dataset ba_2motifs --method size_sc --calculate_all_metrics
python main.py --run_time 10 --dataset ba_2motifs --method gsat_sc --calculate_all_metrics
python main.py --run_time 10 --dataset ba_2motifs --method cal_cr_sc --calculate_all_metrics
```

---

## 📌 Citation

If you find this work useful, please cite us:

```bibtex
@inproceedings{tai2026self,
  title     = {Self-Consistency Improves the Trustworthiness of Self-Interpretable GNNs},
  author    = {Tai, Wenxin and Zhong, Ting and Trajcevski, Goce and Zhou, Fan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

---

## 📬 Contact

If you have questions or suggestions, feel free to reach out via GitHub Issues or email: wxtai [AT] outlook [DOT] com