/* ── Neural network background ───────────────── */
(function () {
  var canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');

  var W, H, particles;
  var COUNT = 70;
  var DIST  = 160;
  var scanning = false;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  function Particle() {
    this.x  = Math.random() * W;
    this.y  = Math.random() * H;
    this.vx = (Math.random() - 0.5) * 0.3;
    this.vy = (Math.random() - 0.5) * 0.3;
    this.r  = Math.random() * 1.8 + 0.6;
    this.hue = Math.random() < 0.65 ? 213 : 267; // blue vs purple
    this.a  = Math.random() * 0.35 + 0.15;
  }

  particles = [];
  for (var i = 0; i < COUNT; i++) particles.push(new Particle());

  function frame() {
    ctx.clearRect(0, 0, W, H);

    var speed = scanning ? 3.5 : 1;

    // update positions
    for (var i = 0; i < particles.length; i++) {
      var p = particles[i];
      p.x += p.vx * speed;
      p.y += p.vy * speed;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
    }

    // draw edges
    for (var i = 0; i < particles.length; i++) {
      for (var j = i + 1; j < particles.length; j++) {
        var dx = particles[i].x - particles[j].x;
        var dy = particles[i].y - particles[j].y;
        var d  = Math.sqrt(dx * dx + dy * dy);
        if (d < DIST) {
          var a = (1 - d / DIST) * (scanning ? 0.2 : 0.09);
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = 'rgba(88,140,255,' + a + ')';
          ctx.lineWidth = 0.7;
          ctx.stroke();
        }
      }
    }

    // draw nodes
    for (var i = 0; i < particles.length; i++) {
      var p = particles[i];
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = 'hsla(' + p.hue + ',90%,70%,' + (scanning ? p.a * 2 : p.a) + ')';
      ctx.fill();
    }

    requestAnimationFrame(frame);
  }
  frame();

  window.startScan = function () {
    scanning = true;
    setTimeout(function () { scanning = false; }, 2800);
  };
})();

/* ── Typewriter subtitle ─────────────────────── */
var _tw = "Detect hate speech, toxicity and offensive content";
var _ti = 0;
function typeWriter() {
  var el = document.getElementById('ai-typing');
  if (!el || _ti >= _tw.length) return;
  el.textContent += _tw[_ti++];
  setTimeout(typeWriter, 30);
}

/* ── Form submit ─────────────────────────────── */
document.getElementById('analyze-form').addEventListener('submit', function () {
  var btn  = document.getElementById('analyze-btn');
  var text = document.getElementById('btn-text');
  var ldr  = document.getElementById('btn-loader');
  var beam = document.getElementById('scan-beam');
  if (text) text.style.display = 'none';
  if (ldr)  ldr.style.display  = 'flex';
  if (btn)  btn.disabled       = true;
  if (beam) beam.classList.add('active');
  if (window.startScan) window.startScan();
});

window.addEventListener('load', typeWriter);
