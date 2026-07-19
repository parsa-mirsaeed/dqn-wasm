import init, { DqnDemo } from '../pkg/dqn_wasm.js';

const log = document.getElementById('log');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function write(msg) {
  log.textContent = msg + "\n" + log.textContent;
}

function draw(state) {
  const [x, , theta] = state;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const cartY = 170;
  const cartX = canvas.width / 2 + x * 80;
  const cartW = 70;
  const cartH = 30;

  ctx.fillStyle = '#333';
  ctx.fillRect(0, cartY + cartH / 2, canvas.width, 2);

  ctx.fillStyle = '#1e88e5';
  ctx.fillRect(cartX - cartW / 2, cartY - cartH / 2, cartW, cartH);

  const poleLen = 90;
  const px = cartX;
  const py = cartY - 8;
  const tipX = px + poleLen * Math.sin(theta);
  const tipY = py - poleLen * Math.cos(theta);

  ctx.strokeStyle = '#e53935';
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(tipX, tipY);
  ctx.stroke();
}

async function main() {
  await init();
  const demo = new DqnDemo();

  draw(demo.get_state());

  document.getElementById('reset').onclick = () => {
    const s = demo.reset();
    draw(s);
    write('Environment reset');
  };

  document.getElementById('train1').onclick = () => {
    const stats = demo.train_episode(300);
    write(JSON.stringify(stats));
    draw(demo.get_state());
  };

  document.getElementById('train50').onclick = async () => {
    for (let i = 0; i < 50; i++) {
      const stats = demo.train_episode(300);
      if ((i + 1) % 10 === 0) write(JSON.stringify(stats));
      await new Promise(r => setTimeout(r, 0));
    }
    draw(demo.get_state());
  };

  document.getElementById('run').onclick = async () => {
    demo.reset();
    for (let t = 0; t < 300; t++) {
      const action = demo.act_greedy();
      const [state, reward, done] = demo.step_env(action);
      draw(state);
      await new Promise(r => setTimeout(r, 30));
      if (done) {
        write(`Run finished at step ${t + 1}, reward=${reward}`);
        break;
      }
    }
  };
}

main();
