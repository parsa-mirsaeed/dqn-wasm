use js_sys::{Array, Math};
use serde::Serialize;
use wasm_bindgen::prelude::*;

const STATE_DIM: usize = 4;
const HIDDEN_DIM: usize = 32;
const ACTION_DIM: usize = 2;

const GRAVITY: f32 = 9.8;
const MASSCART: f32 = 1.0;
const MASSPOLE: f32 = 0.1;
const TOTAL_MASS: f32 = MASSPOLE + MASSCART;
const LENGTH: f32 = 0.5;
const POLEMASS_LENGTH: f32 = MASSPOLE * LENGTH;
const FORCE_MAG: f32 = 10.0;
const TAU: f32 = 0.02;
const X_THRESHOLD: f32 = 2.4;
const THETA_THRESHOLD_RADIANS: f32 = 12.0 * 2.0 * std::f32::consts::PI / 360.0;

fn rand_f32() -> f32 {
    Math::random() as f32
}

fn rand_range(low: f32, high: f32) -> f32 {
    low + (high - low) * rand_f32()
}

#[derive(Clone)]
struct Transition {
    state: [f32; STATE_DIM],
    action: usize,
    reward: f32,
    next_state: [f32; STATE_DIM],
    done: bool,
}

struct ReplayBuffer {
    data: Vec<Transition>,
    capacity: usize,
    position: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self { data: Vec::with_capacity(capacity), capacity, position: 0 }
    }

    fn push(&mut self, t: Transition) {
        if self.data.len() < self.capacity {
            self.data.push(t);
        } else {
            self.data[self.position] = t;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn sample_indices(&self, batch: usize) -> Vec<usize> {
        (0..batch)
            .map(|_| ((rand_f32() * self.data.len() as f32) as usize).min(self.data.len() - 1))
            .collect()
    }
}

#[derive(Clone)]
struct Mlp {
    w1: [[f32; STATE_DIM]; HIDDEN_DIM],
    b1: [f32; HIDDEN_DIM],
    w2: [[f32; HIDDEN_DIM]; ACTION_DIM],
    b2: [f32; ACTION_DIM],
}

impl Mlp {
    fn new() -> Self {
        let mut w1 = [[0.0; STATE_DIM]; HIDDEN_DIM];
        let mut w2 = [[0.0; HIDDEN_DIM]; ACTION_DIM];
        for i in 0..HIDDEN_DIM {
            for j in 0..STATE_DIM {
                w1[i][j] = rand_range(-0.1, 0.1);
            }
        }
        for i in 0..ACTION_DIM {
            for j in 0..HIDDEN_DIM {
                w2[i][j] = rand_range(-0.1, 0.1);
            }
        }
        Self { w1, b1: [0.0; HIDDEN_DIM], w2, b2: [0.0; ACTION_DIM] }
    }

    fn forward(&self, x: &[f32; STATE_DIM]) -> ([f32; HIDDEN_DIM], [f32; HIDDEN_DIM], [f32; ACTION_DIM]) {
        let mut z1 = [0.0; HIDDEN_DIM];
        let mut h1 = [0.0; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            let mut s = self.b1[i];
            for j in 0..STATE_DIM {
                s += self.w1[i][j] * x[j];
            }
            z1[i] = s;
            h1[i] = if s > 0.0 { s } else { 0.0 };
        }
        let mut out = [0.0; ACTION_DIM];
        for a in 0..ACTION_DIM {
            let mut s = self.b2[a];
            for i in 0..HIDDEN_DIM {
                s += self.w2[a][i] * h1[i];
            }
            out[a] = s;
        }
        (z1, h1, out)
    }

    fn predict(&self, x: &[f32; STATE_DIM]) -> [f32; ACTION_DIM] {
        self.forward(x).2
    }

    fn train_single(&mut self, x: &[f32; STATE_DIM], action: usize, target: f32, lr: f32) -> f32 {
        let (z1, h1, q) = self.forward(x);
        let pred = q[action];
        let diff = pred - target;
        let loss = 0.5 * diff * diff;

        let mut grad_out = [0.0; ACTION_DIM];
        grad_out[action] = diff;

        for a in 0..ACTION_DIM {
            for i in 0..HIDDEN_DIM {
                self.w2[a][i] -= lr * grad_out[a] * h1[i];
            }
            self.b2[a] -= lr * grad_out[a];
        }

        let mut grad_h = [0.0; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            grad_h[i] = self.w2[action][i] * diff;
        }

        let mut grad_z1 = [0.0; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            grad_z1[i] = if z1[i] > 0.0 { grad_h[i] } else { 0.0 };
        }

        for i in 0..HIDDEN_DIM {
            for j in 0..STATE_DIM {
                self.w1[i][j] -= lr * grad_z1[i] * x[j];
            }
            self.b1[i] -= lr * grad_z1[i];
        }

        loss
    }
}

#[derive(Clone)]
struct CartPole {
    state: [f32; STATE_DIM],
}

impl CartPole {
    fn new() -> Self {
        let mut env = Self { state: [0.0; STATE_DIM] };
        env.reset();
        env
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
        let x = self.state[0];
        let x_dot = self.state[1];
        let theta = self.state[2];
        let theta_dot = self.state[3];

        let force = if action == 1 { FORCE_MAG } else { -FORCE_MAG };
        let costheta = theta.cos();
        let sintheta = theta.sin();

        let temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS;
        let thetaacc = (GRAVITY * sintheta - costheta * temp)
            / (LENGTH * (4.0 / 3.0 - MASSPOLE * costheta * costheta / TOTAL_MASS));
        let xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

        let x = x + TAU * x_dot;
        let x_dot = x_dot + TAU * xacc;
        let theta = theta + TAU * theta_dot;
        let theta_dot = theta_dot + TAU * thetaacc;

        self.state = [x, x_dot, theta, theta_dot];

        let done = x < -X_THRESHOLD
            || x > X_THRESHOLD
            || theta < -THETA_THRESHOLD_RADIANS
            || theta > THETA_THRESHOLD_RADIANS;

        let reward = if done { 0.0 } else { 1.0 };
        (self.state, reward, done)
    }
}

#[derive(Serialize)]
struct TrainStats {
    episode: usize,
    reward: f32,
    epsilon: f32,
    avg_loss: f32,
}

#[wasm_bindgen]
pub struct DqnDemo {
    env: CartPole,
    online: Mlp,
    target: Mlp,
    replay: ReplayBuffer,
    gamma: f32,
    epsilon: f32,
    epsilon_min: f32,
    epsilon_decay: f32,
    lr: f32,
    batch_size: usize,
    target_update_every: usize,
    steps: usize,
    episode: usize,
    last_reward: f32,
}

#[wasm_bindgen]
impl DqnDemo {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let online = Mlp::new();
        let target = online.clone();
        Self {
            env: CartPole::new(),
            online,
            target,
            replay: ReplayBuffer::new(10_000),
            gamma: 0.99,
            epsilon: 1.0,
            epsilon_min: 0.05,
            epsilon_decay: 0.995,
            lr: 0.001,
            batch_size: 32,
            target_update_every: 200,
            steps: 0,
            episode: 0,
            last_reward: 0.0,
        }
    }

    pub fn reset(&mut self) -> JsValue {
        self.last_reward = 0.0;
        serde_wasm_bindgen::to_value(&self.env.reset()).unwrap()
    }

    pub fn get_state(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.env.state).unwrap()
    }

    pub fn q_values(&self) -> JsValue {
        let q = self.online.predict(&self.env.state);
        serde_wasm_bindgen::to_value(&q).unwrap()
    }

    pub fn act_greedy(&self) -> usize {
        let q = self.online.predict(&self.env.state);
        if q[1] > q[0] { 1 } else { 0 }
    }

    pub fn step_env(&mut self, action: usize) -> JsValue {
        let (s, r, d) = self.env.step(action);
        self.last_reward += r;
        let arr = Array::new();
        arr.push(&serde_wasm_bindgen::to_value(&s).unwrap());
        arr.push(&JsValue::from_f64(r as f64));
        arr.push(&JsValue::from_bool(d));
        arr.into()
    }

    pub fn train_episode(&mut self, max_steps: usize) -> JsValue {
        let mut state = self.env.reset();
        let mut ep_reward = 0.0;
        let mut total_loss = 0.0;
        let mut updates = 0usize;

        for _ in 0..max_steps {
            let action = if rand_f32() < self.epsilon {
                if rand_f32() < 0.5 { 0 } else { 1 }
            } else {
                let q = self.online.predict(&state);
                if q[1] > q[0] { 1 } else { 0 }
            };

            let (next_state, reward, done) = self.env.step(action);
            self.replay.push(Transition { state, action, reward, next_state, done });
            ep_reward += reward;
            state = next_state;
            self.steps += 1;

            if self.replay.len() >= self.batch_size {
                let idxs = self.replay.sample_indices(self.batch_size);
                let mut batch_loss = 0.0;
                for idx in idxs {
                    let t = self.replay.data[idx].clone();
                    let next_q = self.target.predict(&t.next_state);
                    let max_next = next_q[0].max(next_q[1]);
                    let target = if t.done { t.reward } else { t.reward + self.gamma * max_next };
                    batch_loss += self.online.train_single(&t.state, t.action, target, self.lr);
                }
                total_loss += batch_loss / self.batch_size as f32;
                updates += 1;
            }

            if self.steps % self.target_update_every == 0 {
                self.target = self.online.clone();
            }

            if done {
                break;
            }
        }

        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
        self.episode += 1;

        let stats = TrainStats {
            episode: self.episode,
            reward: ep_reward,
            epsilon: self.epsilon,
            avg_loss: if updates > 0 { total_loss / updates as f32 } else { 0.0 },
        };
        serde_wasm_bindgen::to_value(&stats).unwrap()
    }
}
