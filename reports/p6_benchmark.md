# ChaturangZero-X — AI Benchmark Report

_Generated: 2026-09-30 09:31:34_

---

## 1. Training Curves

| Epoch | Train Policy Loss | Train Value Loss | Val Policy Loss | Val Value Loss | Val Top-1 Acc |
|-------|-------------------|-----------------|-----------------|----------------|---------------|
|     1 | 6.7347            | 0.0052            | 5.8635           | 0.0246           | 2.4%        |
|     2 | 5.5871            | 0.0031            | 5.7578           | 0.0243           | 2.7%        |
|     3 | 5.3571            | 0.0028            | 5.7074           | 0.0244           | 3.1%        |

---

## 2. Game Quality Scores (Random Play Baseline)

| Variant | Decisiveness | Tactical Complexity | Material Volatility | Blunder Freq | **Quality Score** |
|---------|-------------|---------------------|--------------------|--------------|--------------------|
| Standard (R1)        | 0.133        | 0.481               | 0.409              | 0.245       | **0.1644**        |
| Capture-Raja (R2)    | 0.067        | 0.464               | 0.465              | 0.263       | **0.1431**        |
| Bare-Raja (R3)       | 0.067        | 0.468               | 0.458              | 0.302       | **0.1376**        |
| No-Stalemate (R4)    | 0.000        | 0.489               | 0.431              | 0.288       | **0.1223**        |

---

## 3. Bayesian Rule Inference Accuracy

**Test accuracy: 32.5%**  
> (% of games where highest-posterior variant matches the true generating variant)


---

## 4. Strategy DNA — Cosine Distance Matrix

> Lower value = more similar strategic fingerprint between the two variants.


| Variant | Standard (R1) | Capture-Raja (R | Bare-Raja (R3) | No-Stalemate (R |
|---------|----------|----------|----------|----------|
| Standard (R1)   |  —       | 0.0020   | 0.0047   | 0.0024   |
| Capture-Raja (R | 0.0020   |  —       | 0.0024   | 0.0018   |
| Bare-Raja (R3)  | 0.0047   | 0.0024   |  —       | 0.0033   |
| No-Stalemate (R | 0.0024   | 0.0018   | 0.0033   |  —       |

---

## 5. Key Findings

- Best game quality variant: **standard**
- Bayesian rule inference accuracy: **32.5%**
- Final val policy loss: **5.7074**
- Final val top-1 accuracy: **3.1%**

---

_ChaturangZero-X — Computational Game Archaeology AI_