import init, { DqnAgent } from './pkg/dqn_wasm.js';

const $ = id => document.getElementById(id);
const envCanvas   = $('env');
const envCtx      = envCanvas.getContext('2d');
const chartCanvas = $('chart');
const chartCtx    = chartCanvas.getContext('2d');
const rewards     = [];

function resizeCanvas(c) {
  c.width  = c.clientWidth  * window.devicePixelRatio;
  c.height = c.clientHeight * window.devicePixelRatio;
}
resizeCanvas(envCanvas);
resizeCanvas(chartCanvas);

function drawEnv(state) {
  const W = envCanvas.width, H = envCanvas.height;
  const dpr = window.devicePixelRatio;
  envCtx.clearRect(0, 0, W, H);
  const [x, , theta] = state;
  const groundY = H * 0.72;
  const cartW   = 80 * dpr, cartH = 28 * dpr;
  const cartX   = W / 2 + x * (W * 0.15);

  envCtx.strokeStyle = '#30363d';
  envCtx.lineWidth   = 2 * dpr;
  envCtx.beginPath();
  envCtx.moveTo(0, groundY + cartH / 2);
  envCtx.lineTo(W, groundY + cartH / 2);
  envCtx.stroke();

  envCtx.fillStyle = '#1f6feb';
  envCtx.beginPath();
  envCtx.roundRect(cartX - cartW / 2, groundY - cartH / 2, cartW, cartH, 5 * dpr);
  envCtx.fill();

  for (const wx of [-cartW * 0.32, cartW * 0.32]) {
    envCtx.beginPath();
    envCtx.arc(cartX + wx, groundY + cartH / 2, 8 * dpr, 0, Math.PI * 2);
    envCtx.fillStyle = '#8b949e';
    envCtx.fill();
  }

  const poleLen = 100 * dpr;
  const px = cartX, py = groundY - cartH / 2;
  const tipX = px + poleLen * Math.sin(theta);
  const tipY = py - poleLen * Math.cos(theta);
  envCtx.strokeStyle = '#f78166';
  envCtx.lineWidth   = 7 * dpr;
  envCtx.lineCap     = 'round';
  envCtx.beginPath();
  envCtx.moveTo(px, py);
  envCtx.lineTo(tipX, tipY);
  envCtx.stroke();

  envCtx.beginPath();
  envCtx.arc(px, py, 5 * dpr, 0, Math.PI * 2);
  envCtx.fillStyle = '#e6edf3';
  envCtx.fill();
}

function drawChart() {
  const W = chartCanvas.width, H = chartCanvas.height;
  const dpr = window.devicePixelRatio;
  chartCtx.clearRect(0, 0, W, H);
  if (rewards.length < 2) return;

  const pad  = { t: 10*dpr, r: 10*dpr, b: 30*dpr, l: 40*dpr };
  const iW   = W - pad.l - pad.r;
  const iH   = H - pad.t - pad.b;
  const maxR = Math.max(...rewards, 1);

  chartCtx.strokeStyle = '#30363d';
  chartCtx.lineWidth   = 1 * dpr;
  chartCtx.beginPath();
  chartCtx.moveTo(pad.l, pad.t);
  chartCtx.lineTo(pad.l, H - pad.b);
  chartCtx.lineTo(W - pad.r, H - pad.b);
  chartCtx.stroke();

  chartCtx.fillStyle = '#8b949e';
  chartCtx.font      = `${9*dpr}px system-ui`;
  chartCtx.textAlign = 'right';
  chartCtx.fillText(maxR.toFixed(0), pad.l - 4*dpr, pad.t + 4*dpr);
  chartCtx.fillText('0', pad.l - 4*dpr, H - pad.b);

  const smoothed = rewards.map((_, i) => {
    const w = rewards.slice(Math.max(0, i - 9), i + 1);
    return w.reduce((a, b) => a + b, 0) / w.length;
  });

  const toX = i => pad.l + (i / (rewards.length - 1)) * iW;
  const toY = v => H - pad.b - (v / maxR) * iH;

  chartCtx.strokeStyle = 'rgba(31,111,235,0.3)';
  chartCtx.lineWidth   = 1 * dpr;
  chartCtx.beginPath();
  rewards.forEach((r, i) => i === 0 ? chartCtx.moveTo(toX(i), toY(r)) : chartCtx.lineTo(toX(i), toY(r)));
  chartCtx.stroke();

  chartCtx.strokeStyle = '#58a6ff';
  chartCtx.lineWidth   = 2.5 * dpr;
  chartCtx.beginPath();
  smoothed.forEach((r, i) => i === 0 ? chartCtx.moveTo(toX(i), toY(r)) : chartCtx.lineTo(toX(i), toY(r)));
  chartCtx.stroke();
}

function log(msg) {
  const el = $('log');
  el.textContent = msg + '\n' + el.textContent;
}

function updateStats(stats, info) {
  $('sEp').textContent  = info.episode;
  $('sRew').textContent = stats.reward.toFixed(1);
  $('sEps').textContent = stats.epsilon.toFixed(3);
  $('sBuf').textContent = info.buffer_size;
}

async function main() {
  await init();
  let agent = new DqnAgent();
  drawEnv(agent.get_state());

  $('btnReset').onclick = () => {
    agent = new DqnAgent();
    rewards.length = 0;
    drawChart();
    drawEnv(agent.get_state());
    $('sEp').textContent  = '0';
    $('sRew').textContent = '-';
    $('sEps').textContent = '1.000';
    $('sBuf').textContent = '0';
    log('Agent reset.');
  };

  async function trainN(n) {
    [$('btn1'), $('btn10'), $('btn100')].forEach(b => b.disabled = true);
    for (let i = 0; i < n; i++) {
      const stats = agent.train_episode(500);
      const info  = agent.info();
      rewards.push(stats.reward);
      updateStats(stats, info);
      drawChart();
      if (i % Math.max(1, Math.floor(n / 5)) === 0 || i === n - 1) {
        log(`ep=${stats.episode} reward=${stats.reward.toFixed(1)} eps=${stats.epsilon.toFixed(3)} loss=${stats.avg_loss.toFixed(4)}`);
        drawEnv(agent.get_state());
        await new Promise(r => setTimeout(r, 0));
      }
    }
    [$('btn1'), $('btn10'), $('btn100')].forEach(b => b.disabled = false);
  }

  $('btn1').onclick   = () => trainN(1);
  $('btn10').onclick  = () => trainN(10);
  $('btn100').onclick = () => trainN(100);

  $('btnRun').onclick = async () => {
    agent.reset();
    let t = 0;
    const run = async () => {
      if (t++ >= 500) { log('Run ended (500 step limit).'); return; }
      const action = agent.act_greedy();
      const [state, , done] = agent.step_env(action);
      drawEnv(state);
      if (done) { log(`Greedy run ended at step ${t}.`); return; }
      requestAnimationFrame(run);
    };
    requestAnimationFrame(run);
  };
}

main();
