# Self-Consistency Improves the Trustworthiness of Self-Interpretable GNNs

**💻 Official implementation of our ICLR submission: Self-Consistency Improves the Trustworthiness of Self-Interpretable GNNs**

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
├── model.py        # GNN backbone (GIN/GCN)
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

Datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1RaOKbWABerHfea_sJZGIbXSy0FcOzK0O?usp=sharing) (from our baseline [1]), place all datasets (e.g., `ba_2motifs`, `benzene`, `mr`, `mutag`) in the `datasets/` folder.  

[1]: Redundancy Undermines the Trustworthiness of Self-Interpretable GNNs, ICML, 2025

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
We provide **pretrained model checkpoints** for quick reproduction:

```bash
python main.py --run_time 10 --dataset ba_2motifs --method size_sc --calculate_all_metrics
python main.py --run_time 10 --dataset ba_2motifs --method gsat_sc --calculate_all_metrics
python main.py --run_time 10 --dataset ba_2motifs --method cal_cr_sc --calculate_all_metrics
```

---