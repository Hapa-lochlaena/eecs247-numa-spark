"""
numa_spark_ml.py
================
EECS247 26Spring - Yu Zhang - Project 2 final report code.

Two stages, both on Apache Spark:

  Stage A (Spark MLlib).  Train logistic-regression classifiers the canonical
  way (https://spark.apache.org/docs/latest/ml-classification-regression.html
  #logistic-regression):
      * binary LR  on sample_libsvm_data.txt          (face-detection analog)
      * multinomial LR on the multiclass sample data   (keyword-spotting analog)
  Each trained model IS a fully-connected layer: its coefficient matrix has
  shape (numClasses x numFeatures), i.e. a weight matrix W.

  Stage B (Spark RDD + Spark SQL).  Take the trained W shapes (plus the
  keyword-spotting layer dimensions reported in Bang et al., ISSCC 2017,
  Fig. 14.7.6) and run the NUMA-vs-UMA on-chip memory-energy analysis from the
  proposal, as a distributed design-space sweep (network x tile x precision)
  with a Spark SQL window query for the optimal tile, plus the drowsy-mode
  leakage model.

This connects a standard Spark MLlib workflow (the classifier) to the project
topic (the energy of *deploying* that classifier under a NUMA memory hierarchy).

Run:  spark-submit numa_spark_ml.py        (or: python numa_spark_ml.py)
"""

import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)

# ---------------------------------------------------------------------------
# NUMA / UMA energy model (same ratios as the proposal & paper Fig. 14.7.5)
# ---------------------------------------------------------------------------
ACCESS_ENERGY = {1: 1.0, 2: 1.4, 3: 1.9, 4: 2.5}   # L1=1.0 ... L4=2.5
UMA_LEVEL = 2.0
PERIPH_LEAK_SHARE = 0.91                            # paper Fig. 14.7.5
N_BANKS = 16                                        # 4 levels x 4 banks / PE


def _bytes(elems, bits):
    return elems * bits / 8.0


def numa_energy(N, M, T, bits):
    """NUMA access energy for an N x M FCL, T rows per tile."""
    n_tiles = int(np.ceil(N / T))
    e = 0.0
    for i in range(n_tiles):
        rows = min(T, N - i * T)
        e += _bytes(M, bits) * ACCESS_ENERGY[1]        # input  -> L1 (reused)
        e += _bytes(rows * M, bits) * ACCESS_ENERGY[4] # weights-> L4
        e += _bytes(rows, 32) * ACCESS_ENERGY[2]       # output -> L2
    return e


def uma_energy(N, M, bits):
    return (_bytes(N * M, bits) + _bytes(N * M, bits) + _bytes(N, 32)) * UMA_LEVEL


def best_numa(N, M, bits):
    """Best NUMA energy over all tile sizes (optimum is the single tile)."""
    return min(numa_energy(N, M, T, bits) for T in (1, max(1, N // 2), N))


def drowsy_reduction(active_fraction):
    return (1.0 - active_fraction) * PERIPH_LEAK_SHARE


# ---------------------------------------------------------------------------
# Convolutional layers: the paper supports conv by FFT, transforming it into
# a matrix-multiplication. We model that equivalent GEMM and run the same
# NUMA-vs-UMA accounting. A conv layer with Cin->Cout channels, KxK kernel,
# and P = Hout*Wout output positions is P matrix-vector products that share
# one weight matrix W of shape (Cout x Cin*K*K).
#   NUMA  : input feature map loaded once into L1 and reused across the KxK
#           window; weights streamed from L4; outputs accumulated in L2.
#   UMA   : no locality, so the input is touched in im2col-expanded form
#           (Cin*K*K per position), weights once, all at the uniform cost.
# ---------------------------------------------------------------------------
def conv_numa(Cin, Cout, K, P, bits):
    inp = Cin * P                  # compact feature map, loaded once (reused)
    wgt = Cout * Cin * K * K       # kernel weights from L4
    out = Cout * P                 # output activations (32-bit accumulation)
    return (_bytes(inp, bits) * ACCESS_ENERGY[1]
            + _bytes(wgt, bits) * ACCESS_ENERGY[4]
            + _bytes(out, 32) * ACCESS_ENERGY[2])


def conv_uma(Cin, Cout, K, P, bits):
    inp = Cin * K * K * P          # im2col-expanded input, no reuse
    wgt = Cout * Cin * K * K       # weights once
    out = Cout * P
    return (_bytes(inp, bits) + _bytes(wgt, bits) + _bytes(out, 32)) * UMA_LEVEL


def find_sample(name):
    base = os.path.join(os.path.dirname(__import__("pyspark").__file__), "data", "mllib")
    hits = glob.glob(os.path.join(base, name))
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
def main():
    spark = (SparkSession.builder
             .appName("NUMA-DLA-Spark-MLlib")
             .master("local[*]")
             .config("spark.sql.shuffle.partitions", "16")
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    print(">> Spark", spark.version, "| cores =", sc.defaultParallelism)
    R = {}

    # ===================================================================
    # STAGE A : Spark MLlib logistic regression  (the classifiers / FCLs)
    # ===================================================================
    # --- A1: binary LR  (face-detection analog) ------------------------
    bin_path = find_sample("sample_libsvm_data.txt")
    dfb = spark.read.format("libsvm").load(bin_path)
    lr_b = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.0)
    mb = lr_b.fit(dfb)
    pb = mb.transform(dfb)
    auc = BinaryClassificationEvaluator(metricName="areaUnderROC").evaluate(pb)
    acc_b = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(pb)
    Nb, Mb = int(mb.numClasses), int(mb.numFeatures)
    print(f">> [MLlib] binary LR  : feats={Mb}, classes={Nb}, "
          f"AUC={auc:.3f}, acc={acc_b:.3f}")

    # --- A2: multinomial LR  (keyword-spotting analog) -----------------
    mc_path = find_sample("sample_multiclass_classification_data.txt")
    dfm = spark.read.format("libsvm").load(mc_path)
    lr_m = LogisticRegression(maxIter=30, regParam=0.1, family="multinomial")
    mm = lr_m.fit(dfm)
    pm = mm.transform(dfm)
    acc_m = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(pm)
    Nm, Mm = int(mm.numClasses), int(mm.numFeatures)
    print(f">> [MLlib] multinom LR: feats={Mm}, classes={Nm}, acc={acc_m:.3f}")
    print(f">>          coefficientMatrix shape = "
          f"{mm.coefficientMatrix.numRows} x {mm.coefficientMatrix.numCols}")

    R["mllib"] = {
        "binary":     {"features": Mb, "classes": Nb,
                       "auc": round(auc, 4), "accuracy": round(acc_b, 4),
                       "fcl": f"{Nb}x{Mb}"},
        "multiclass": {"features": Mm, "classes": Nm,
                       "accuracy": round(acc_m, 4),
                       "fcl": f"{Nm}x{Mm}"},
    }

    # FCL shapes from the trained models + the paper's keyword-spotting layers
    trained_layers = [("Binary LR (libsvm)", Nb, Mb),
                      ("Multinomial LR", Nm, Mm)]
    paper_layers = [("KWS layer-1", 384, 408),
                    ("KWS layer-2", 384, 384),
                    ("KWS layer-3", 24, 384)]

    bits = 8
    trained_e = [(name, N, M,
                  best_numa(N, M, bits), uma_energy(N, M, bits))
                 for (name, N, M) in (trained_layers + paper_layers)]
    R["deploy_energy"] = [
        {"layer": n, "N": N, "M": M,
         "numa_e": round(ne, 2), "uma_e": round(ue, 2),
         "savings_pct": round((1 - ne / ue) * 100, 2)}
        for (n, N, M, ne, ue) in trained_e]
    print(">> [Deploy] NUMA vs UMA energy of trained / paper FCLs (8-bit):")
    for d in R["deploy_energy"]:
        print(f"     {d['layer']:<20} {d['N']}x{d['M']:<5} "
              f"savings={d['savings_pct']:.1f}%")

    # ===================================================================
    # STAGE B : distributed NUMA design-space sweep (RDD + Spark SQL)
    # ===================================================================
    networks = [(64, 64), (128, 128), (256, 256), (384, 408),
                (512, 512), (768, 768), (1024, 1024)]
    precisions = [6, 8, 12, 16]
    configs = []
    for (N, M) in networks:
        tiles = sorted(set(list(range(1, N + 1, max(1, N // 80))) + [N]))
        for T in tiles:
            for b in precisions:
                configs.append((N, M, T, b))
    print(f">> design-space points = {len(configs):,}")

    grid = sc.parallelize(configs, numSlices=sc.defaultParallelism)

    def evaluate(cfg):
        N, M, T, b = cfg
        ne = numa_energy(N, M, T, b); ue = uma_energy(N, M, b)
        return (N, M, T, b, float(ne), float(ue),
                float((1.0 - ne / ue) * 100.0))

    swept = grid.map(evaluate).cache()
    print(f">> evaluated {swept.count():,} configurations on Spark")
    df = spark.createDataFrame(
        swept, ["N", "M", "T", "bits", "numa_e", "uma_e", "savings"]).cache()

    # Exp 1: 384x408 tile sweep (8-bit)
    layer = (df.filter((F.col("N") == 384) & (F.col("bits") == 8))
               .orderBy("T").toPandas())
    R["exp1"] = {"peak_savings_pct": round(float(layer["savings"].max()), 2),
                 "uma_e": float(layer["uma_e"].iloc[0])}

    # Exp 2: optimal tile per network via Spark SQL window
    w = Window.partitionBy("N", "M", "bits").orderBy(F.desc("savings"))
    best = (df.filter(F.col("bits") == 8)
              .withColumn("rk", F.row_number().over(w))
              .filter(F.col("rk") == 1).orderBy("N").toPandas())
    R["exp2"] = {"network": [f"{int(r.N)}x{int(r.M)}" for r in best.itertuples()],
                 "best_tile": best["T"].tolist(),
                 "savings_pct": best["savings"].round(2).tolist()}

    # Exp 4: precision sweep on 384x408
    prec = (df.filter(F.col("N") == 384).groupBy("bits")
              .agg(F.max("savings").alias("best")).orderBy("bits").toPandas())
    R["exp_precision"] = {"bits": prec["bits"].tolist(),
                          "best_savings_pct": prec["best"].round(2).tolist()}

    # Exp 5: drowsy mode
    drowsy = (sc.parallelize(range(1, N_BANKS + 1))
                .map(lambda a: (a, round(drowsy_reduction(a / N_BANKS) * 100, 2)))
                .collect())
    drowsy.sort()
    R["exp_drowsy"] = {"active_banks": [d[0] for d in drowsy],
                       "reduction_pct": [d[1] for d in drowsy],
                       "schedule_active_banks": 6,
                       "schedule_reduction_pct":
                           round(drowsy_reduction(6 / N_BANKS) * 100, 2)}

    # Exp 6: convolutional layers via FFT / matrix-multiplication path
    # (name, Cin, Cout, K, Hout, Wout) for a small face-detection-style CNN
    conv_layers = [("conv1", 1, 16, 3, 32, 32),
                   ("conv2", 16, 32, 3, 16, 16),
                   ("conv3", 32, 64, 3, 8, 8),
                   ("conv4", 64, 64, 3, 4, 4)]

    def eval_conv(c):
        name, Cin, Cout, K, Ho, Wo = c
        P = Ho * Wo
        ne = conv_numa(Cin, Cout, K, P, 8)
        ue = conv_uma(Cin, Cout, K, P, 8)
        return (name, Cin, Cout, K, P, float(ne), float(ue),
                float((1.0 - ne / ue) * 100.0))

    conv = sc.parallelize(conv_layers).map(eval_conv).collect()
    conv.sort(key=lambda r: r[0])
    R["exp_conv"] = [{"layer": r[0], "Cin": r[1], "Cout": r[2], "K": r[3],
                      "P": r[4], "numa_e": round(r[5], 2), "uma_e": round(r[6], 2),
                      "savings_pct": round(r[7], 2)} for r in conv]
    print(">> [Conv] NUMA vs UMA for conv layers (as matrix-mult, 8-bit):")
    for d in R["exp_conv"]:
        print(f"     {d['layer']:<7} {d['Cin']:>3}->{d['Cout']:<3} K{d['K']} "
              f"P={d['P']:<5} savings={d['savings_pct']:.1f}%")

    # ===================================================================
    # Plots
    # ===================================================================
    # Fig 0: MLlib results + deploy energy of trained FCLs
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.bar(["Binary LR\n(AUC)", "Binary LR\n(acc)", "Multinom LR\n(acc)"],
           [auc, acc_b, acc_m], color=["#2E75B6", "#5B9BD5", "#27AE60"], width=0.6)
    a1.set_ylim(0, 1.05); a1.set_ylabel("Score")
    a1.set_title("Spark MLlib logistic-regression quality")
    for i, v in enumerate([auc, acc_b, acc_m]):
        a1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    a1.grid(alpha=0.3, axis="y")

    names = [d["layer"] for d in R["deploy_energy"]]
    ne = [d["numa_e"] / 1e3 for d in R["deploy_energy"]]
    ue = [d["uma_e"] / 1e3 for d in R["deploy_energy"]]
    x = np.arange(len(names))
    a2.bar(x - 0.2, ue, 0.4, label="UMA", color="#C0392B")
    a2.bar(x + 0.2, ne, 0.4, label="NUMA", color="#2E75B6")
    a2.set_xticks(x); a2.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    a2.set_ylabel("Access energy (x10^3 a.u.)")
    a2.set_title("Deploy energy of trained / paper FCLs (8-bit)")
    a2.set_yscale("log"); a2.legend(); a2.grid(alpha=0.3, which="both", axis="y")
    fig.tight_layout(); fig.savefig("fig0_mllib.png", dpi=150); plt.close(fig)

    # Fig 1: tile sweep
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(layer["T"], layer["numa_e"] / 1e6, color="#2E75B6", lw=2, label="NUMA")
    a1.axhline(R["exp1"]["uma_e"] / 1e6, color="#C0392B", lw=2, ls="--", label="UMA")
    a1.set_xlabel("Tile size (rows/tile)"); a1.set_ylabel("Access energy (x10^6 a.u.)")
    a1.set_title("NUMA vs UMA (384x408 FCL, 8-bit)"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(layer["T"], layer["savings"], color="#27AE60", lw=2, label="NUMA savings")
    a2.axhline(40, color="gray", lw=1.5, ls=":", label="Paper (~40%)")
    a2.set_xlabel("Tile size (rows/tile)"); a2.set_ylabel("Savings vs UMA (%)")
    a2.set_ylim(0, 60); a2.set_title("Energy savings vs tile size"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("fig1_tile_sweep.png", dpi=150); plt.close(fig)

    # Fig 2: network scaling
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    sizes = best["N"].tolist()
    a1.plot(sizes, best["numa_e"] / 1e6, "o-", color="#2E75B6", lw=2, label="NUMA")
    a1.plot(sizes, best["uma_e"] / 1e6, "s--", color="#C0392B", lw=2, label="UMA")
    a1.set_xlabel("Layer dim N (NxN)"); a1.set_ylabel("Access energy (x10^6 a.u.)")
    a1.set_yscale("log"); a1.set_title("Energy vs network size (8-bit)")
    a1.legend(); a1.grid(alpha=0.3, which="both")
    a2.plot(sizes, best["savings"], "o-", color="#27AE60", lw=2)
    a2.axhline(40, color="gray", lw=1.5, ls=":", label="Paper (~40%)")
    a2.set_xlabel("Layer dim N (NxN)"); a2.set_ylabel("Savings vs UMA (%)")
    a2.set_ylim(0, 60); a2.set_title("NUMA savings vs network size"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("fig2_network_scaling.png", dpi=150); plt.close(fig)

    # Fig 3: precision + drowsy
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.bar([str(b) for b in prec["bits"]], prec["best"], color="#8E44AD", width=0.6)
    a1.axhline(40, color="gray", lw=1.5, ls=":", label="Paper (~40%)")
    a1.set_xlabel("Weight precision (bits)"); a1.set_ylabel("Best NUMA savings (%)")
    a1.set_ylim(0, 60); a1.set_title("Precision x NUMA savings (384x408)")
    a1.legend(); a1.grid(alpha=0.3, axis="y")
    ab = [d[0] for d in drowsy]; rd = [d[1] for d in drowsy]
    a2.plot(ab, rd, "o-", color="#E67E22", lw=2, label="Modelled")
    a2.axhline(54, color="gray", lw=1.5, ls=":", label="Paper (54%)")
    a2.axvline(6, color="#2E75B6", lw=1.2, ls="--", label="FCL schedule (6/16)")
    a2.set_xlabel("Active banks (of 16)"); a2.set_ylabel("Leakage reduction (%)")
    a2.set_title("Drowsy-mode leakage reduction"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("fig3_precision_drowsy.png", dpi=150); plt.close(fig)

    # Fig 4: convolutional layers
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    cn = [d["layer"] for d in R["exp_conv"]]
    cne = [d["numa_e"] / 1e3 for d in R["exp_conv"]]
    cue = [d["uma_e"] / 1e3 for d in R["exp_conv"]]
    csv = [d["savings_pct"] for d in R["exp_conv"]]
    xx = np.arange(len(cn))
    a1.bar(xx - 0.2, cue, 0.4, label="UMA", color="#C0392B")
    a1.bar(xx + 0.2, cne, 0.4, label="NUMA", color="#2E75B6")
    a1.set_xticks(xx); a1.set_xticklabels(cn)
    a1.set_ylabel("Access energy (x10^3 a.u.)")
    a1.set_title("Conv layers: NUMA vs UMA (as matrix-mult, 8-bit)")
    a1.set_yscale("log"); a1.legend(); a1.grid(alpha=0.3, which="both", axis="y")
    a2.bar(cn, csv, color="#27AE60", width=0.6)
    a2.axhline(37.4, color="gray", lw=1.5, ls=":", label="FCL savings (~37%)")
    a2.set_ylabel("Energy savings vs UMA (%)")
    a2.set_title("Conv-layer savings depend on shape")
    a2.set_ylim(0, 70); a2.legend(); a2.grid(alpha=0.3, axis="y")
    for i, v in enumerate(csv):
        a2.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig("fig4_conv.png", dpi=150); plt.close(fig)

    with open("results.json", "w") as fh:
        json.dump(R, fh, indent=2)
    print(">> wrote results.json + 5 figures")
    spark.stop()


if __name__ == "__main__":
    main()
