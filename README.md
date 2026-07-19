# DQN in Rust + WebAssembly

A browser-runnable implementation of **Deep Q-Networks (DQN)** inspired by the paper **"Playing Atari with Deep Reinforcement Learning"** by Mnih et al. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602).

This repo intentionally implements the **core logic faithfully** while using a smaller benchmark environment, **CartPole**, so the algorithm can run directly in the browser through **Rust + wasm-bindgen + JavaScript**.

## What is implemented

Core DQN ingredients from the paper:

- Q-network updated by temporal-difference learning
- Experience replay buffer
- Fixed target network
- Epsilon-greedy exploration
- Mini-batch updates with MSE TD loss

## Deliberate simplifications

The original paper learns from raw Atari pixels using a convolutional neural network and frame stacking. This implementation uses:

- low-dimensional state vectors instead of image input
- a small MLP instead of a CNN
- CartPole dynamics instead of Atari

These changes preserve the algorithmic idea while making the project practical for a public browser demo.

## Repo layout

- `src/lib.rs` — DQN agent, replay buffer, neural net, CartPole env, wasm exports
- `www/index.html` — browser UI
- `www/main.js` — JS integration with the WASM module
- `Cargo.toml` — Rust package config

## Run locally

### Prerequisites

- Rust stable
- `wasm-pack`
- a static file server such as `python -m http.server`

### Build

```bash
wasm-pack build --target web
```

### Serve

From the repository root:

```bash
python -m http.server 8000
```

Then open:

- [http://localhost:8000/www/](http://localhost:8000/www/)

## Notes on accuracy

This repository is a **reference implementation of DQN-style learning**, not a claim of exact Atari reproduction. The implementation keeps the learning rule and architectural ideas clear and auditable in Rust.

## Reference

```bibtex
@article{mnih2013playing,
  title={Playing Atari with Deep Reinforcement Learning},
  author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Graves, Alex and Antonoglou, Ioannis and Wierstra, Daan and Riedmiller, Martin},
  journal={arXiv preprint arXiv:1312.5602},
  year={2013}
}
```
