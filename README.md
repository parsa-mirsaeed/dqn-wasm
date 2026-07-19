# DQN · Rust + WebAssembly

[![CI](https://github.com/parsa-mirsaeed/dqn-wasm/actions/workflows/ci.yml/badge.svg)](https://github.com/parsa-mirsaeed/dqn-wasm/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-GitHub%20Pages-blue)](https://parsa-mirsaeed.github.io/dqn-wasm/)
[![arXiv](https://img.shields.io/badge/paper-arXiv%3A1312.5602-b31b1b)](https://arxiv.org/abs/1312.5602)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A **research-grade browser demo** of **Deep Q-Networks (DQN)** from:

> V. Mnih et al., *"Playing Atari with Deep Reinforcement Learning"*, arXiv:1312.5602, 2013
> (Nature version: DOI [10.1038/nature14236](https://doi.org/10.1038/nature14236))

Full **Algorithm 1** from the paper is implemented in **pure Rust**, compiled to
**WebAssembly** with `wasm-bindgen`, and rendered in the browser with vanilla JavaScript.

---

## Live demo

**[parsa-mirsaeed.github.io/dqn-wasm](https://parsa-mirsaeed.github.io/dqn-wasm/)**

---

## What this implements

All core components of Algorithm 1 (paper §4):

| Component | Paper | Code |
|---|---|---|
| Experience replay | §4.1 | `ReplayBuffer` in `src/lib.rs` |
| Target Q-network | Algorithm 1, line 10 | `self.target = self.online.clone()` |
| epsilon-greedy exploration | Eq. 1 | `if randf() < self.epsilon` |
| TD target y_j | Algorithm 1, line 11 | `let y = if done { r } else { r + gamma*max Q_hat }` |
| Mini-batch SGD | Algorithm 1, line 11 | `sgd_step()` with manual backprop |

See [`PAPER.md`](PAPER.md) for a **line-by-line mapping** of Algorithm 1 to Rust code,
and a hyperparameter correspondence table.

---

## Repo structure

```
|- src/lib.rs              # DQN agent, CartPole env, MLP, replay buffer, unit tests
|- www/
|   |- index.html          # Browser UI
|   |- main.js             # JS glue layer
|   `- pkg/                # wasm-pack output (after build)
|- Cargo.toml
|- PAPER.md                # Paper reference + algorithm mapping table
|- .github/workflows/ci.yml
`- LICENSE
```

---

## Build & run locally

### Prerequisites

```bash
# Rust stable
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

### Build WASM

```bash
wasm-pack build --target web --out-dir www/pkg
```

### Serve

```bash
python -m http.server 8000 --directory www
# or: npx serve www
```

Open: http://localhost:8000

### Run unit tests (native)

```bash
cargo test --lib
```

---

## CI & GitHub Pages

On every push to `main`, GitHub Actions:
1. Installs Rust + wasm-pack
2. Runs `cargo test --lib` (5 unit tests)
3. Builds WASM in release mode
4. Deploys `www/` to GitHub Pages

---

## Deliberate adaptations

The original paper uses Atari 2600 pixel frames + a CNN + frame stacking.
This repo uses **CartPole** + **MLP** for practical browser execution.
The algorithmic logic (replay, target net, TD learning) is unchanged.
See [`PAPER.md`](PAPER.md) for full details.

---

## Reference

```bibtex
@article{mnih2013playing,
  title   = {Playing Atari with Deep Reinforcement Learning},
  author  = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and
             Graves, Alex and Antonoglou, Ioannis and Wierstra, Daan and
             Riedmiller, Martin},
  journal = {arXiv preprint arXiv:1312.5602},
  year    = {2013},
  url     = {https://arxiv.org/abs/1312.5602}
}
```
