# EECS247 Project 2 — NUMA-Aware Memory Hierarchy on Apache Spark

Final project for **EECS247 (26Spring)**, by **Yu Zhang**.

This project extends the Project-2 proposal (a simulation of the Non-Uniform
Memory Access hierarchy from Bang et al., *ISSCC 2017*, "A 288µW Programmable
Deep-Learning Processor … Using Non-Uniform Memory Hierarchy for Mobile
Intelligence") into a two-stage **Apache Spark** program.

## What it does

**Stage A — Spark MLlib logistic regression.** Trains classifiers the canonical
[MLlib way](https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression):
a binary LR on the bundled `sample_libsvm_data.txt` (face-detection analog) and a
multinomial LR on the multiclass sample (keyword-spotting analog). Each trained
model's coefficient matrix is a fully-connected layer `W` of shape
`numClasses × numFeatures`.

**Stage B — Spark RDD + Spark SQL.** Feeds the trained `W` shapes (plus the
keyword-spotting layer dimensions reported in the paper) into a distributed
NUMA-vs-UMA on-chip memory-energy analysis: a design-space sweep over
`network × tile × precision` as an RDD `map`, a Spark SQL window query for the
optimal tile per layer, and a drowsy-mode leakage model.

## Run

```bash
pip install -r requirements.txt
python numa_spark_ml.py          # local[*]
# or on a cluster:
spark-submit numa_spark_ml.py
```

Outputs: `results.json` and four figures (`fig0_mllib.png` … `fig3_precision_drowsy.png`).

## Files

| File | Description |
|------|-------------|
| `numa_spark_ml.py` | Main Spark program (Stages A + B) |
| `requirements.txt` | Python dependencies |
| `results.json` | Numeric results (generated) |

## Reference libraries

This project models accelerator memory energy analytically. The standard
open-source tools in this space, used here as references for the energy-modeling
methodology:

- **Timeloop** — DNN accelerator modeling & mapping: https://github.com/NVlabs/timeloop
- **Accelergy** — architecture-level energy estimation: https://github.com/Accelergy-Project/accelergy
- **SCALE-Sim** — systolic-array accelerator simulator (FC/GEMM, memory accesses): https://github.com/scalesim-project/scale-sim-v2
- **Apache Spark MLlib** — classification/regression: https://github.com/apache/spark

## Reference

S. Bang et al., "A 288µW Programmable Deep-Learning Processor with 270KB On-Chip
Weight Storage Using Non-Uniform Memory Hierarchy for Mobile Intelligence,"
*ISSCC*, pp. 250–251, 2017.
