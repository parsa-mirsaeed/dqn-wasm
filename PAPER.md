# Paper Reference & Algorithm Mapping

## Citation

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

Nature journal version (2015):

```bibtex
@article{mnih2015humanlevel,
  title   = {Human-level control through deep reinforcement learning},
  author  = {Mnih, Volodymyr and others},
  journal = {Nature},
  volume  = {518},
  pages   = {529--533},
  year    = {2015},
  doi     = {10.1038/nature14236}
}
```

## Algorithm 1 -> Code Mapping

Each line of **Algorithm 1** from the paper mapped to Rust code in `src/lib.rs`.

| Algorithm 1 (paper)                                                        | Rust location                                     |
|----------------------------------------------------------------------------|---------------------------------------------------|
| Initialise replay memory D with capacity N                                 | `ReplayBuffer::new(BUFFER_CAP)` -- const `10_000` |
| Initialise Q-network with random weights theta                             | `Mlp::new()` (He initialisation)                  |
| Initialise target Q-network theta- <- theta                                | `target = online.clone()`                         |
| For each episode                                                           | `train_episode()` outer loop                      |
| Preprocess phi_t (frame stack + grayscale)                                 | **Not applicable** -- CartPole state is 4-vector  |
| epsilon-greedy: with prob epsilon select random action                     | `if randf() < self.epsilon` block                 |
| otherwise a_t = argmax_a Q(phi_t, a; theta)                                | `if q[1] > q[0]` branch                          |
| Execute a_t, observe r_t, phi_{t+1}                                        | `self.env.step(a)`                                |
| Store (phi_t, a_t, r_t, phi_{t+1}) in D                                   | `self.replay.push(Transition { ... })`            |
| Sample random mini-batch of size B from D                                  | `self.replay.sample(BATCH)`                       |
| y_j = r_j if terminal, else r_j + gamma * max Q_hat(phi_{j+1}; theta-)    | `let y = if t.done { t.r } else { t.r + ... }`   |
| Perform SGD step on (y_j - Q(phi_j, a_j; theta))^2                        | `self.online.sgd_step(&t.s, t.a, y, LR)`         |
| Every C steps reset theta- <- theta                                        | `if self.steps % TARGET_SYNC == 0` block          |

## Hyperparameter correspondence

| Paper (Nature 2015, Table 1) | This implementation | Value  |
|------------------------------|---------------------|--------|
| Replay memory size N         | `BUFFER_CAP`        | 10 000 |
| Mini-batch size              | `BATCH`             | 64     |
| Discount factor gamma        | `GAMMA`             | 0.99   |
| Target sync period C         | `TARGET_SYNC`       | 300    |
| Learning rate                | `LR`                | 5e-4   |
| Initial epsilon              | `EPS_START`         | 1.0    |
| Final epsilon                | `EPS_END`           | 0.05   |
| Optimiser                    | SGD (manual)        | SGD    |

## Deliberate adaptations

| Original paper               | This repo                         | Reason                              |
|------------------------------|-----------------------------------|-------------------------------------|
| Atari 2600 pixels (84x84x4)  | CartPole 4D state vector          | WASM / browser -- no image I/O      |
| Convolutional Q-network      | 2-layer MLP (4 -> 64 -> 2)        | Zero heavy deps in Rust/WASM        |
| RMSProp optimiser            | SGD (manual backprop)             | Avoids allocations in WASM context  |
| 1M replay entries            | 10k                               | Browser memory limits               |
| epsilon annealed 1M frames   | Exponential decay x0.995/episode  | Faster convergence for CartPole     |
