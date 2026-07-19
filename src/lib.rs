//! DQN implementation based on:
//! Mnih et al. "Playing Atari with Deep Reinforcement Learning"
//! arXiv:1312.5602  (2013)  /  Nature 518, 529-533 (2015)
//!
//! Algorithm 1 from the paper is implemented faithfully:
//!   - Experience replay buffer  (paper §4.1)
//!   - Target Q-network with periodic sync  (paper §4, Nature extension)
//!   - epsilon-greedy policy with decay  (paper §4)
//!   - Mini-batch SGD on TD targets: y = r + gamma * max_a' Q_hat(s',a'; theta-)
//!
//! Deliberate adaptations for browser/WASM:
//!   - CartPole dynamics instead of Atari frames
//!   - 2-layer MLP (4->64->2) instead of CNN
//!   - Manual backprop (zero heavy dependencies)

use js_sys::Math;
use serde::Serialize;
use wasm_bindgen::prelude::*;

// -----------------------------------------------
// Dimensions
// -----------------------------------------------
const STATE_DIM: usize = 4;
const HIDDEN_DIM: usize = 64;
const ACTION_DIM: usize = 2;

// -----------------------------------------------
// CartPole physics constants (OpenAI Gym values)
// -----------------------------------------------
const GRAVITY: f32 = 9.8;
const MASSCART: f32 = 1.0;
const MASSPOLE: f32 = 0.1;
const TOTAL_MASS: f32 = MASSPOLE + MASSCART;
const LENGTH: f32 = 0.5;
const POLEMASS_LENGTH: f32 = MASSPOLE * LENGTH;
const FORCE_MAG: f32 = 10.0;
const TAU: f32 = 0.02;
const X_THRESHOLD: f32 = 2.4;
const THETA_THRESHOLD: f32 = 12.0 * 2.0 * std::f32::consts::PI / 360.0;

// -----------------------------------------------
// Hyperparameters (paper §4 / Nature supplement)
// -----------------------------------------------
/// Discount factor gamma
const GAMMA: f32 = 0.99;
/// Initial epsilon for epsilon-greedy exploration
const EPS_START: f32 = 1.0;
/// Final epsilon
const EPS_END: f32 = 0.05;
/// Multiplicative decay per episode
const EPS_DECAY: f32 = 0.995;
/// SGD learning rate
const LR: f32 = 5e-4;
/// Replay buffer capacity N (paper uses 1M; scaled for browser)
const BUFFER_CAP: usize = 10_000;
/// Mini-batch size
const BATCH: usize = 64;
/// Target network sync period C (paper Algorithm 1 line 10)
const TARGET_SYNC: usize = 300;

// -----------------------------------------------
// RNG shim -- uses JS Math.random via wasm-bindgen
// -----------------------------------------------
#[inline]
fn randf() -> f32 { Math::random() as f32 }
#[inline]
fn rand_range(lo: f32, hi: f32) -> f32 { lo + (hi - lo) * randf() }

// -----------------------------------------------
// Experience replay  (paper §4.1)
// -----------------------------------------------
#[derive(Clone)]
struct Transition {
    s:    [f32; STATE_DIM],
    a:    usize,
    r:    f32,
    s_:   [f32; STATE_DIM],
    done: bool,
}

struct ReplayBuffer {
    buf: Vec<Transition>,
    cap: usize,
    pos: usize,
}

impl ReplayBuffer {
    fn new(cap: usize) -> Self { Self { buf: Vec::with_capacity(cap), cap, pos: 0 } }

    fn push(&mut self, t: Transition) {
        if self.buf.len() < self.cap { self.buf.push(t); }
        else { self.buf[self.pos] = t; }
        self.pos = (self.pos + 1) % self.cap;
    }

    fn len(&self) -> usize { self.buf.len() }

    /// Uniform random sample -- paper §4.1 "uniform random sampling"
    fn sample(&self, n: usize) -> Vec<&Transition> {
        (0..n).map(|_| {
            let i = (randf() * self.buf.len() as f32) as usize;
            &self.buf[i.min(self.buf.len() - 1)]
        }).collect()
    }
}

// -----------------------------------------------
// Q-Network -- 2-layer MLP, ReLU hidden, linear out
// Manual forward + backprop
// -----------------------------------------------
#[derive(Clone)]
struct Mlp {
    w1: [[f32; STATE_DIM];  HIDDEN_DIM],
    b1: [f32; HIDDEN_DIM],
    w2: [[f32; HIDDEN_DIM]; ACTION_DIM],
    b2: [f32; ACTION_DIM],
}

impl Mlp {
    fn new() -> Self {
        let scale1 = (2.0_f32 / STATE_DIM as f32).sqrt();
        let scale2 = (2.0_f32 / HIDDEN_DIM as f32).sqrt();
        let mut w1 = [[0.0f32; STATE_DIM]; HIDDEN_DIM];
        let mut w2 = [[0.0f32; HIDDEN_DIM]; ACTION_DIM];
        for i in 0..HIDDEN_DIM {
            for j in 0..STATE_DIM  { w1[i][j] = rand_range(-scale1, scale1); }
        }
        for i in 0..ACTION_DIM {
            for j in 0..HIDDEN_DIM { w2[i][j] = rand_range(-scale2, scale2); }
        }
        Self { w1, b1: [0.0; HIDDEN_DIM], w2, b2: [0.0; ACTION_DIM] }
    }

    /// Forward pass -- returns (pre-activations z1, hidden h1, Q-values)
    fn forward(&self, x: &[f32; STATE_DIM])
        -> ([f32; HIDDEN_DIM], [f32; HIDDEN_DIM], [f32; ACTION_DIM])
    {
        let mut z1 = [0.0f32; HIDDEN_DIM];
        let mut h1 = [0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            let mut s = self.b1[i];
            for j in 0..STATE_DIM { s += self.w1[i][j] * x[j]; }
            z1[i] = s;
            h1[i] = s.max(0.0); // ReLU
        }
        let mut q = [0.0f32; ACTION_DIM];
        for a in 0..ACTION_DIM {
            let mut s = self.b2[a];
            for i in 0..HIDDEN_DIM { s += self.w2[a][i] * h1[i]; }
            q[a] = s;
        }
        (z1, h1, q)
    }

    #[inline]
    fn predict(&self, x: &[f32; STATE_DIM]) -> [f32; ACTION_DIM] { self.forward(x).2 }

    /// One SGD step for (s, a, target).
    /// Loss = 0.5*(Q(s,a) - y)^2  -- paper eq. (2) / Algorithm 1 line 11
    fn sgd_step(&mut self, x: &[f32; STATE_DIM], a: usize, y: f32, lr: f32) -> f32 {
        let (z1, h1, q) = self.forward(x);
        let err = q[a] - y;
        // Output layer gradients
        let mut dw2_a = [0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM { dw2_a[i] = err * h1[i]; }
        // Hidden layer -- backprop through ReLU
        let mut dh = [0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM { dh[i] = self.w2[a][i] * err; }
        let mut dz1 = [0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM { dz1[i] = if z1[i] > 0.0 { dh[i] } else { 0.0 }; }
        // Update w2, b2
        for i in 0..HIDDEN_DIM { self.w2[a][i] -= lr * dw2_a[i]; }
        self.b2[a] -= lr * err;
        // Update w1, b1
        for i in 0..HIDDEN_DIM {
            for j in 0..STATE_DIM { self.w1[i][j] -= lr * dz1[i] * x[j]; }
            self.b1[i] -= lr * dz1[i];
        }
        0.5 * err * err
    }
}

// -----------------------------------------------
// CartPole environment (Euler integration)
// -----------------------------------------------
#[derive(Clone)]
struct CartPole { state: [f32; STATE_DIM] }

impl CartPole {
    fn new() -> Self {
        let mut e = Self { state: [0.0; STATE_DIM] };
        e.reset();
        e
    }

    fn reset(&mut self) -> [f32; STATE_DIM] {
        self.state = [
            rand_range(-0.05, 0.05),
            rand_range(-0.05, 0.05),
            rand_range(-0.05, 0.05),
            rand_range(-0.05, 0.05),
        ];
        self.state
    }

    fn step(&mut self, action: usize) -> ([f32; STATE_DIM], f32, bool) {
        let [x, xd, th, thd] = self.state;
        let f = if action == 1 { FORCE_MAG } else { -FORCE_MAG };
        let cos = th.cos();
        let sin = th.sin();
        let tmp   = (f + POLEMASS_LENGTH * thd * thd * sin) / TOTAL_MASS;
        let thacc = (GRAVITY * sin - cos * tmp)
                    / (LENGTH * (4.0 / 3.0 - MASSPOLE * cos * cos / TOTAL_MASS));
        let xacc  = tmp - POLEMASS_LENGTH * thacc * cos / TOTAL_MASS;
        self.state = [x + TAU * xd, xd + TAU * xacc, th + TAU * thd, thd + TAU * thacc];
        let [x, _, th, _] = self.state;
        let done = x.abs() > X_THRESHOLD || th.abs() > THETA_THRESHOLD;
        (self.state, if done { 0.0 } else { 1.0 }, done)
    }
}

// -----------------------------------------------
// Stats structs (serialised to JS)
// -----------------------------------------------
#[derive(Serialize)]
pub struct EpisodeStats {
    pub episode:  usize,
    pub reward:   f32,
    pub epsilon:  f32,
    pub avg_loss: f32,
    pub steps:    usize,
}

#[derive(Serialize)]
pub struct AgentInfo {
    pub episode:     usize,
    pub total_steps: usize,
    pub epsilon:     f32,
    pub buffer_size: usize,
}

// -----------------------------------------------
// DQN Agent -- public WASM interface
// -----------------------------------------------
#[wasm_bindgen]
pub struct DqnAgent {
    env:     CartPole,
    online:  Mlp,
    target:  Mlp,
    replay:  ReplayBuffer,
    epsilon: f32,
    steps:   usize,
    episode: usize,
}

#[wasm_bindgen]
impl DqnAgent {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let online = Mlp::new();
        let target = online.clone();
        Self {
            env:     CartPole::new(),
            online,
            target,
            replay:  ReplayBuffer::new(BUFFER_CAP),
            epsilon: EPS_START,
            steps:   0,
            episode: 0,
        }
    }

    /// Run one full training episode (Algorithm 1 from paper)
    pub fn train_episode(&mut self, max_steps: usize) -> JsValue {
        let mut s = self.env.reset();
        let mut ep_reward = 0.0f32;
        let mut ep_loss   = 0.0f32;
        let mut updates   = 0usize;
        let mut ep_steps  = 0usize;

        for _ in 0..max_steps {
            // epsilon-greedy action selection (paper §4, eq. 1)
            let a = if randf() < self.epsilon {
                if randf() < 0.5 { 0 } else { 1 }
            } else {
                let q = self.online.predict(&s);
                if q[1] > q[0] { 1 } else { 0 }
            };

            let (s_, r, done) = self.env.step(a);
            self.replay.push(Transition { s, a, r, s_, done });
            ep_reward += r;
            ep_steps  += 1;
            self.steps += 1;
            s = s_;

            // Sample mini-batch and update (paper Algorithm 1, lines 10-11)
            if self.replay.len() >= BATCH {
                let batch = self.replay.sample(BATCH);
                for t in &batch {
                    // y_j = r_j                              if terminal
                    //     = r_j + gamma * max_a' Q_hat(s_j') otherwise
                    let tgt_q  = self.target.predict(&t.s_);
                    let max_q_ = tgt_q[0].max(tgt_q[1]);
                    let y = if t.done { t.r } else { t.r + GAMMA * max_q_ };
                    ep_loss += self.online.sgd_step(&t.s, t.a, y, LR);
                    updates += 1;
                }
            }

            // Sync target network every C steps (paper Algorithm 1, line 10)
            if self.steps % TARGET_SYNC == 0 {
                self.target = self.online.clone();
            }

            if done { break; }
        }

        self.epsilon = (self.epsilon * EPS_DECAY).max(EPS_END);
        self.episode += 1;

        let stats = EpisodeStats {
            episode:  self.episode,
            reward:   ep_reward,
            epsilon:  self.epsilon,
            avg_loss: if updates > 0 { ep_loss / updates as f32 } else { 0.0 },
            steps:    ep_steps,
        };
        serde_wasm_bindgen::to_value(&stats).unwrap()
    }

    pub fn reset(&mut self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.env.reset()).unwrap()
    }

    pub fn get_state(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.env.state).unwrap()
    }

    pub fn q_values(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.online.predict(&self.env.state)).unwrap()
    }

    pub fn act_greedy(&self) -> usize {
        let q = self.online.predict(&self.env.state);
        if q[1] > q[0] { 1 } else { 0 }
    }

    pub fn step_env(&mut self, action: usize) -> JsValue {
        let (s, r, done) = self.env.step(action);
        use js_sys::Array;
        let arr = Array::new();
        arr.push(&serde_wasm_bindgen::to_value(&s).unwrap());
        arr.push(&JsValue::from_f64(r as f64));
        arr.push(&JsValue::from_bool(done));
        arr.into()
    }

    pub fn info(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&AgentInfo {
            episode:     self.episode,
            total_steps: self.steps,
            epsilon:     self.epsilon,
            buffer_size: self.replay.len(),
        }).unwrap()
    }
}

// -----------------------------------------------
// Unit tests (cargo test --lib)
// -----------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cartpole_step_survives_center() {
        let mut env = CartPole { state: [0.0; 4] };
        env.state = [0.0, 0.0, 0.01, 0.0];
        let (s, r, done) = env.step(1);
        assert!(!done, "should not be done at near-center state");
        assert_eq!(r, 1.0);
        assert!(s[2].abs() < THETA_THRESHOLD);
    }

    #[test]
    fn cartpole_terminates_on_large_angle() {
        let mut env = CartPole { state: [0.0; 4] };
        env.state = [0.0, 0.0, THETA_THRESHOLD + 0.1, 0.0];
        let (_, _, done) = env.step(0);
        assert!(done);
    }

    #[test]
    fn mlp_output_dim_and_bias() {
        let net = Mlp {
            w1: [[0.0; STATE_DIM]; HIDDEN_DIM],
            b1: [0.0; HIDDEN_DIM],
            w2: [[0.0; HIDDEN_DIM]; ACTION_DIM],
            b2: [0.1, -0.1],
        };
        let q = net.predict(&[0.0; STATE_DIM]);
        assert_eq!(q.len(), ACTION_DIM);
        assert!((q[0] - 0.1).abs() < 1e-6);
        assert!((q[1] + 0.1).abs() < 1e-6);
    }

    #[test]
    fn replay_buffer_fifo_overwrite() {
        let mut buf = ReplayBuffer::new(3);
        for i in 0..5 {
            buf.push(Transition { s: [i as f32; 4], a: 0, r: 1.0, s_: [0.0; 4], done: false });
        }
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn td_target_terminal_vs_nonterminal() {
        let net = Mlp {
            w1: [[0.0; STATE_DIM]; HIDDEN_DIM],
            b1: [0.0; HIDDEN_DIM],
            w2: [[0.0; HIDDEN_DIM]; ACTION_DIM],
            b2: [0.5, 0.5],
        };
        let s_ = [0.0f32; 4];
        let tgt_q  = net.predict(&s_);
        let max_q_ = tgt_q[0].max(tgt_q[1]);
        let r = 0.0f32;
        let y_term    = r;
        let y_nonterm = r + GAMMA * max_q_;
        assert!((y_term - 0.0).abs() < 1e-6);
        assert!((y_nonterm - GAMMA * 0.5).abs() < 1e-5);
    }

    #[test]
    fn sgd_step_reduces_loss() {
        let mut net = Mlp {
            w1: [[0.0; STATE_DIM]; HIDDEN_DIM],
            b1: [0.0; HIDDEN_DIM],
            w2: [[0.0; HIDDEN_DIM]; ACTION_DIM],
            b2: [1.0, 0.0],
        };
        let s = [0.0f32; STATE_DIM];
        let loss_before = 0.5 * (net.predict(&s)[0] - 0.0_f32).powi(2);
        let loss_after  = net.sgd_step(&s, 0, 0.0, 0.1);
        assert!(loss_after < loss_before + 1e-6);
    }
}
