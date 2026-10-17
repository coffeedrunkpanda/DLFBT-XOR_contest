# XOR Contest

Train a neural network with **2 inputs and 1 output** to solve XOR in **as few epochs as possible**, and find out which design choices (topology, initialization, activation, cost function, optimizer) make it converge faster.

- 📓 Notebook: [`DLFBT_Valvezan_Anne_XOR_contest.ipynb`](DLFBT_Valvezan_Anne_XOR_contest.ipynb)
- 📄 Report (PDF export): [`DLFBT_Valvezan_Anne_XOR_contest.pdf`](DLFBT_Valvezan_Anne_XOR_contest.pdf)

## Approach

1. **Theory:** why a single neuron can't solve XOR (it isn't linearly separable), and why one hidden layer with at least 2 neurons is needed.
2. **Baseline:** 2-2-1 network, tanh hidden layer, sigmoid output, binary cross-entropy, Glorot normal initialization, SGD (lr = 0.5), full batch.
3. **Convergence criterion:** the error counts as minimized when the **MSE between predictions and targets drops below 0.01**. Training stops at that epoch, and the epoch count is the score. There is no validation set: the 4 XOR points are the entire input space.
4. **Experiments 1–5 (ablation):** change **one setting at a time** from the baseline: hidden activation, output layer + loss, initialization, topology, and optimizer / learning rate.
5. **Experiment 6 (cumulative):** combine the best options step by step, re-testing each group on top of the current best configuration.

## Results

| Experiment | Best option | Epochs | Speed-up |
|---|---|---|---|
| Baseline | 2-2-1, tanh, sigmoid + BCE, SGD 0.5 | 231 | 1.0× |
| 1 · Hidden activation | ReLU | 185 | 1.2× |
| 2 · Output layer and loss | linear output + MSE | 201 | 1.1× |
| 3 · Initialization | Glorot uniform | 199 | 1.2× |
| 4 · Topology | 2 hidden layers [8, 8] | 68 | 3.4× |
| 5 · Optimizer / learning rate | Adam, lr = 0.1 | 39 | 5.9× |
| **6 · Combined** | **2-8-8-1, ReLU, He normal, sigmoid + BCE, Adam 0.2** | **10** | **23×** |

The largest gains came from the **optimizer / learning rate** and the **topology** (wider and deeper than the theoretical minimum). Activation, loss and initialization mattered less for speed, but bad choices (sigmoid hidden layer, zero or badly scaled initialization) slowed training down a lot or stopped it completely.

All experiments use a single seed (3), so small differences between runs are within noise. See the notebook for the full discussion and limitations.

## Run it yourself

Tested with **Python 3.11**.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open the notebook and run all cells (about 5 minutes on a laptop CPU). TensorFlow's deterministic mode is enabled, so the results are reproducible.

**PDF report:** set `REPORT = True` in the first code cell, run all cells, save, then:

```bash
playwright install chromium   # once
jupyter nbconvert --to webpdf DLFBT_Valvezan_Anne_XOR_contest.ipynb
```

## Files

| Path | Content |
|---|---|
| `DLFBT_Valvezan_Anne_XOR_contest.ipynb` | Main notebook: theory, experiments, analysis |
| `DLFBT_Valvezan_Anne_XOR_contest.pdf` | PDF export of the notebook |
| `results/summary.csv` | One row per training run (configuration changes, epochs, status, final MSE, …) |
| `results/final_comparison.csv` | Baseline, best run of each experiment, final combined network |
| `results/runs.json` | Every run in full: configuration, metrics and training curves |
| `requirements.txt` | Python dependencies |

## Author

**Anne Valvezan** · [GitHub @coffeedrunkpanda](https://github.com/coffeedrunkpanda)

XOR contest assignment for the course DLFBT, part of the
[Master in Deep Learning for Audio and Visual Artificial Intelligence](https://www.uam.es/uam/master-universitario-aprendizaje-profundo-tratamiento-senales-audio-video) 
at Universidad Autónoma de Madrid, 2026.
