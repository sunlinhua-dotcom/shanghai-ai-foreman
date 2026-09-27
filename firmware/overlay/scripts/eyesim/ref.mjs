// The approved mock-up's "fine" eye engine (widget xiaozhi_eyes_motion_polish_compare,
// v4), ported line for line from its JavaScript so the firmware engine can be
// checked frame by frame. Changes from the widget, all made to both sides:
//   - deterministic: no random glances, blinks, micro-saccades or sparkles;
//   - the face stays at y = 120 (no captions);
//   - glow cut off at 18 px (widget: 15) and a wider render radius, so the
//     cut-off, not the radius, decides where glow ends;
//   - pixels are rounded instead of truncated.
// Writes raw 240x240 RGB888 frames at 60 fps.
//
//   node ref.mjs OUT.rgb

import fs from 'node:fs';

const TAU = Math.PI * 2;
const cl = (v, a, b) => (v < a ? a : v > b ? b : v);
const hex = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const eio = u => (u < .5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2), eout = u => 1 - Math.pow(1 - u, 3);
function sp(p, v, to, w, z, dt) { v += (-2 * z * w * v - w * w * (p - to)) * dt; return [p + v * dt, v]; }
function seg(px, py, ax, ay, bx, by) { const pax = px - ax, pay = py - ay, bax = bx - ax, bay = by - ay, bb = bax * bax + bay * bay, h = bb > 1e-9 ? cl((pax * bax + pay * bay) / bb, 0, 1) : 0; return Math.hypot(pax - bax * h, pay - bay * h); }
function box(px, py, hx, hy, r) { const qx = Math.abs(px) - hx + r, qy = Math.abs(py) - hy + r; return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r; }
function arcD(px, py, s, c, ra, rb) { px = Math.abs(px); return (c * px > s * py ? Math.hypot(px - s * ra, py - c * ra) : Math.abs(Math.hypot(px, py) - ra)) - rb; }
function heart(x, y) { x = Math.abs(x); if (y + x > 1) return Math.hypot(x - .25, y - .75) - Math.SQRT2 / 4; const m = .5 * Math.max(x + y, 0); return Math.sqrt(Math.min(x * x + (y - 1) * (y - 1), (x - m) * (x - m) + (y - m) * (y - m))) * Math.sign(x - y); }
function star5(px, py, r, rf) { const k1x = .809016994375, k1y = -.587785252292; px = Math.abs(px); let d = 2 * Math.max(k1x * px + k1y * py, 0); px -= d * k1x; py -= d * k1y; d = 2 * Math.max(-k1x * px + k1y * py, 0); px += d * k1x; py -= d * k1y; px = Math.abs(px); py -= r; const bx = -rf * k1y, by = rf * k1x - 1, h = cl((px * bx + py * by) / (bx * bx + by * by), 0, r); return Math.hypot(px - bx * h, py - by * h) * Math.sign(py * bx - px * by); }
const s2l = c => { c /= 255; return c <= .04045 ? c / 12.92 : Math.pow((c + .055) / 1.055, 2.4); };
const l2s = c => cl((c <= .0031308 ? 12.92 * c : 1.055 * Math.pow(Math.max(c, 0), 1 / 2.4) - .055) * 255, 0, 255);
function toOK(c) { const r = s2l(c[0]), g = s2l(c[1]), b = s2l(c[2]), l = Math.cbrt(.4122214708 * r + .5363325363 * g + .0514459929 * b), m = Math.cbrt(.2119034982 * r + .6806995451 * g + .1073969566 * b), s = Math.cbrt(.0883024619 * r + .2817188376 * g + .6299787005 * b); return [.2104542553 * l + .793617785 * m - .0040720468 * s, 1.9779984951 * l - 2.428592205 * m + .4505937099 * s, .0259040371 * l + .7827717662 * m - .808675766 * s]; }
function fromOK(o) { const l = Math.pow(o[0] + .3963377774 * o[1] + .2158037573 * o[2], 3), m = Math.pow(o[0] - .1055613458 * o[1] - .0638541728 * o[2], 3), s = Math.pow(o[0] - .0894841775 * o[1] - 1.291485548 * o[2], 3); return [l2s(4.0767416621 * l - 3.3077115913 * m + .2309699292 * s), l2s(-1.2684380046 * l + 2.6097574011 * m - .3413193965 * s), l2s(-.0041960863 * l - .7034186147 * m + 1.707614701 * s)]; }
const HS = Math.sin(1.15), HC = Math.cos(1.15), TS = Math.sin(2), TC = Math.cos(2), CL1 = Math.hypot(14, 14), CL2 = Math.hypot(27, 30);
const META = {
  idle: { p: ['#9BEBFF', '#2F7BFF'] },
  happy: { p: ['#FFE870', '#FF5FA2'] },
  love: { p: ['#FFA3C8', '#FF2D6F'] },
  star: { p: ['#FFF6B0', '#FFAA1E'] },
  think: { p: ['#F0C8FF', '#8B4DFF'] },
  done: { p: ['#B8FFC8', '#1FC85A'], drop: [-20, 1, 6.5], draw: 1 },
  fail: { p: ['#FFB4A6', '#FF3B30'], drop: [-17, -17, 6.5], draw: 1 },
  scare: { p: ['#FFFFFF', '#9FD8FF'], tr: 'fast' },
};
const DROP = {};
for (const k in META) { const M = META[k]; M.c = [hex(M.p[0]), hex(M.p[1])]; M.o = [toOK(M.c[0]), toOK(M.c[1])]; const d = M.drop || [0, 0, 22]; DROP[k] = (x, y) => Math.hypot(x - d[0], y - d[1]) - d[2]; }
const DIR = { tx: 0, ty: 0, picked: -1, zoning: false };
function direct(now, key) {
  if (key === 'idle') { DIR.tx = 0; DIR.ty = 0; }   // deterministic: look ahead, never zone out
  else { DIR.zoning = false; DIR.tx = Math.sin(now / 1000 * .7) * 3; DIR.ty = Math.cos(now / 1000 * .9) * 2; }
}
const nrm = v => { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; };
const Ln = nrm([-.5, -.6, .62]), Hn = nrm([Ln[0], Ln[1], Ln[2] + 1]), BEV = 9, GLW = 18, RB = 80, RB2 = RB * RB;

const buf = new Uint8Array(240 * 240 * 3), LUT = new Float32Array(192);
const E = { t: 0, key: 'idle', enter: 0, blink: 0, lid: 0, lidV: 0, wd: 1, beat: 0, rc: 1, rs: 0, ta: TS, tb: TC, sc: 1, ss: 0, pk: 1, pkV: 0, kx: 1, ky: 1, yo: 0, shx: 0, en: 1, cy: 120, cb: META.idle.c[1].slice(), oa: META.idle.o[0].slice(), ob: META.idle.o[1].slice(), prog: { done: [1, 1], fail: [1, 1] }, pop: { done: [0, 0], fail: [0, 0] } };
const S = { gx: 0, gy: 0, ex: 0, ey: 0, evx: 0, evy: 0, fx: 0, fy: 0, fvx: 0, fvy: 0, mx: 0, my: 0 };
function checkD(x, y, p, w) { const L = p * (CL1 + CL2); if (L <= CL1) { const u = L / CL1; return seg(x, y, -20, 1, -20 + 14 * u, 1 + 14 * u) - w; } const u = (L - CL1) / CL2; return Math.min(seg(x, y, -20, 1, -6, 15), seg(x, y, -6, 15, -6 + 27 * u, 15 - 30 * u)) - w; }
function crossD(x, y, p, w) { const u1 = cl(p * 2, 0, 1); let d = seg(x, y, -17, -17, -17 + 34 * u1, -17 + 34 * u1) - w; if (p > .5) { const u2 = cl(p * 2 - 1, 0, 1); d = Math.min(d, seg(x, y, 17, -17, 17 - 34 * u2, -17 + 34 * u2) - w * cl((p - .5) * 8, 0, 1)); } return d; }
const SH = {
  idle: (x, y) => { const hw = 35, hh = 42, cz = hh * .44, k = E.blink, y0 = -hh - 1 + E.lid * 2 * hh, ya = y0 + (cz - 1.5 - y0) * k, yl = hh + 1 + (cz + 1.5 - hh - 1) * k; return Math.max(box(x / E.wd, y, hw, hh, 17), ya - y, y - yl); },
  happy: (x, y) => arcD(x, 14 - y, HS, HC, 26, 7),
  love: (x, y) => { const S2 = 50; return heart(x / S2, -y / S2 + .55) * S2; },
  star: (x, y) => star5(x * E.sc - y * E.ss, -(x * E.ss + y * E.sc), 32, .48) - 3,
  think: (x, y, s) => { const sn = E.rs * s; return arcD(x * E.rc - y * sn, x * sn + y * E.rc, E.ta, E.tb, 22, 5.5); },
  done: (x, y, s) => { const i = s < 0 ? 0 : 1; return checkD(x, y, E.prog.done[i], 6.5 * (1 + 1.2 * E.pop.done[i])); },
  fail: (x, y, s) => { const i = s < 0 ? 0 : 1; return crossD(x, y, E.prog.fail[i], 6.5 * (1 + 1.2 * E.pop.fail[i])); },
  scare: (x, y) => Math.hypot(x, y) - 34,
};
const ME = [0, 1].map(() => ({ A: SH.idle, B: SH.idle, drop: DROP.idle, ph: 2, t0: 0, e: 1, q: 1, qv: 0, pure: true, dep: 0, tr: 'drop', via: false, draw: false, glint: -1 }));
function snap(m) {
  if (m.ph === 2 && m.pure) return [m.B, 0];
  if (m.dep >= 2) return [m.ph === 1 ? (m.e < .5 ? m.A : m.drop) : (m.draw || m.q >= .5 ? m.B : (m.via ? m.drop : m.A)), 0];
  const A = m.A, B = m.B, dr = m.drop, ph = m.ph, e = m.e, q = m.q, via = m.via, dw = m.draw;
  return [(x, y, s) => { if (ph === 1) { const a = A(x, y, s); return a + (dr(x, y) - a) * e; } const b = B(x, y, s); if (dw) return b; const a = via ? dr(x, y) : A(x, y, s); return a + (b - a) * q; }, m.dep + 1];
}
function Dm(m, x, y, s) {
  if (m.ph === 1) { const a = m.A(x, y, s); return a + (m.drop(x, y) - a) * m.e; }
  const b = m.B(x, y, s); if (m.pure || m.draw) return b;
  const a = m.via ? m.drop(x, y) : m.A(x, y, s); return a + (b - a) * m.q;
}
function set(k, now) {
  const Mk = META[k];
  E.key = k; E.enter = now;
  const tr = Mk.tr || 'drop';
  for (let i = 0; i < 2; i++) { const m = ME[i], r = snap(m); m.A = r[0]; m.dep = r[1]; m.B = SH[k]; m.drop = DROP[k]; m.t0 = now + (i === 1 && tr !== 'fast' ? 55 : 0); m.e = 0; m.q = 0; m.qv = 0; m.pure = false; m.tr = tr; m.via = tr === 'drop'; m.draw = !!Mk.draw && tr === 'drop'; m.ph = tr === 'drop' ? 1 : 2; m.glint = -1; }
  if (E.prog[k]) E.prog[k] = [0, 0];
  if (k === 'scare') { E.pk = .78; E.pkV = 0; }
}
function step(now, dt) {
  const t = now / 1000, h = dt / 2; E.t = t;
  for (let i = 0; i < 2; i++) {
    const m = ME[i];
    if (m.ph === 1) { const u = (now - m.t0) / 160; m.e = eio(cl(u, 0, 1)); if (u >= 1) { m.ph = 2; m.q = 0; m.qv = 0; } }
    else if (now >= m.t0 && !m.pure) { const w = m.tr === 'fast' ? 26 : m.tr === 'slow' ? 6 : 15, z = m.tr === 'fast' ? .36 : m.tr === 'slow' ? .95 : .58; for (let j = 0; j < 2; j++) { const r = sp(m.q, m.qv, 1, w, z, h); m.q = r[0]; m.qv = r[1]; } m.pure = Math.abs(m.q - 1) < .002 && Math.abs(m.qv) < .01; if (m.pure) { m.q = 1; m.qv = 0; } if (m.via && m.glint < 0 && m.q > .92) m.glint = now; }
    if (m.draw) { E.prog[E.key][i] = m.ph === 2 ? cl(m.q, 0, 1) : 0; E.pop[E.key][i] = Math.max(0, m.q - 1); }
  }
  const Mk = META[E.key], kc = 1 - Math.exp(-dt * 6);
  E.oa = mix(E.oa, Mk.o[0], kc); E.ob = mix(E.ob, Mk.o[1], kc); E.cb = fromOK(E.ob);
  for (let i = 0; i < 64; i++) { const v = i / 63, c = fromOK(mix(E.oa, E.ob, v * v * (3 - 2 * v))); LUT[i * 3] = c[0]; LUT[i * 3 + 1] = c[1]; LUT[i * 3 + 2] = c[2]; }
  E.blink = 0;   // deterministic: no blinks
  let ax = DIR.tx, ay = DIR.ty;
  if (E.key === 'idle' && now - DIR.picked < 120) { ax *= .9; ay *= .9; }
  for (let j = 0; j < 2; j++) {
    let r = sp(S.ex, S.evx, ax + S.mx, 34, .78, h); S.ex = r[0]; S.evx = r[1]; r = sp(S.ey, S.evy, ay + S.my, 34, .78, h); S.ey = r[0]; S.evy = r[1]; r = sp(S.fx, S.fvx, ax, 5.5, .95, h); S.fx = r[0]; S.fvx = r[1]; r = sp(S.fy, S.fvy, ay, 5.5, .95, h); S.fy = r[0]; S.fvy = r[1];
    r = sp(E.lid, E.lidV, DIR.zoning && E.key === 'idle' ? .26 : 0, 16, .85, h); E.lid = r[0]; E.lidV = r[1];
  }
  S.gx = .62 * S.ex + .38 * S.fx; S.gy = .62 * S.ey + .38 * S.fy;
  E.wd = 1 + .12 * E.blink;
  const bp = (t * 1.25) % 1; E.beat = Math.exp(-((bp - .1) ** 2) / .002) + .6 * Math.exp(-((bp - .32) ** 2) / .002);
  { const per = 1.6, nn = Math.floor(t / per), f = cl((t - nn * per) / .55, 0, 1), a = (nn + eio(f)) * TAU / 5; E.sc = Math.cos(a); E.ss = Math.sin(a); }
  { const th = t * 4.2 + .9 * Math.sin(t * 2.4), ap = .35 + 1.9 * (.5 - .5 * Math.cos(t * 2.4)); E.rc = Math.cos(th); E.rs = Math.sin(th); E.ta = Math.sin(ap); E.tb = Math.cos(ap); }
  let kx = 1, ky = 1, yo = 0, en = 1, shx = 0; const age = now - E.enter;
  if (E.key === 'idle') { const b = .012 * Math.sin(t * 1.5); kx += b; ky += b; en += .06 * Math.sin(t * 1.5); }
  else if (E.key === 'happy') { const s = Math.abs(Math.sin(t * 3.6)), c = Math.pow(1 - s, 3); yo = -5 * Math.pow(s, .8); kx += .05 * c; ky -= .06 * c; }
  else if (E.key === 'love') { kx += .05 * E.beat; ky += .05 * E.beat; en += .2 * E.beat; }
  else if (E.key === 'star') { const tw = .05 * Math.pow(Math.max(0, Math.sin(t * 2.2)), 6); kx += tw; ky += tw; en += .2; }
  else if (E.key === 'scare') { const r = sp(E.pk, E.pkV, 1, 24, .32, dt); E.pk = r[0]; E.pkV = r[1]; kx *= E.pk; ky *= E.pk; yo = -6 * Math.exp(-Math.pow((age - 70) / 110, 2)); shx = Math.sin(t * 38) * 3.2 * Math.exp(-age / 320); en += .25; }
  const vx = Math.min(1, Math.abs(S.evx) / 700), vy = Math.min(1, Math.abs(S.evy) / 700); kx *= 1 + .06 * vx - .04 * vy; ky *= 1 + .06 * vy - .04 * vx;
  E.kx = kx; E.ky = ky; E.yo = yo; E.en = en; E.shx = shx;
}
function render(now) {
  buf.fill(0);
  const cb = E.cb, gxn = cl(S.gx / 30, -1, 1), km = Math.min(E.kx, E.ky);
  for (const s of [-1, 1]) {
    const m = ME[s < 0 ? 0 : 1];
    const cx = 120 + s * 52 + S.gx + E.shx, cy = E.cy + S.gy + E.yo, ps = 1 + .06 * gxn * s, sx = E.kx * ps, sy = E.ky * ps, sm = km * ps;
    const D = (x, y) => Dm(m, x / sx, y / sy, s) * sm;
    let gA = 0, gP = 0; if (m.glint >= 0) { const u = (now - m.glint) / 520; if (u < 1) { gA = Math.sin(Math.PI * u); gP = -60 + 120 * eio(u); } }
    const x0 = Math.max(0, Math.floor(cx - RB)), x1 = Math.min(239, Math.ceil(cx + RB)), y0 = Math.max(0, Math.floor(cy - RB)), y1 = Math.min(239, Math.ceil(cy + RB));
    for (let py = y0; py <= y1; py++) {
      const ly = py + .5 - cy;
      for (let px = x0; px <= x1; px++) {
        const lx = px + .5 - cx; if (lx * lx + ly * ly > RB2) continue;
        const d = D(lx, ly); if (d > GLW) continue;
        let r, g, b;
        if (d >= .5) { const gl = (Math.exp(-d / 3.2) * .34 + Math.exp(-d / 9) * .2) * E.en; r = cb[0] * gl; g = cb[1] * gl; b = cb[2] * gl; }
        else {
          let nx = 0, ny = 0, nz = 1, k = 0;
          if (-d < BEV) { const dx = D(lx + 1, ly) - d, dy = D(lx, ly + 1) - d, gl = Math.hypot(dx, dy) || 1; k = Math.min(1, 1 + d / BEV); nx = dx / gl * k; ny = dy / gl * k; nz = Math.sqrt(Math.max(0, 1 - k * k)); }
          const nl = Math.max(0, nx * Ln[0] + ny * Ln[1] + nz * Ln[2]), nh = Math.max(0, nx * Hn[0] + ny * Hn[1] + nz * Hn[2]), vi = cl(((ly + 40) * .7875) | 0, 0, 63) * 3;
          const br = .34 + .72 * nl;
          let ad = 230 * Math.pow(nh, 60) + 36 * Math.pow(nh, 8) + 38 * Math.exp(-((lx + 11) * (lx + 11) / 300 + (ly + 17) * (ly + 17) / 90));
          if (gA > 0) { const q = lx * .7 + ly * .7 - gP; ad += 140 * gA * Math.exp(-q * q / 30); }
          r = LUT[vi] * br + ad; g = LUT[vi + 1] * br + ad; b = LUT[vi + 2] * br + ad;
          if (k > .5) { const rm = Math.pow(k, 4) * .35; r += cb[0] * rm; g += cb[1] * rm; b += cb[2] * rm; }
          const cov = cl(.5 - d, 0, 1);
          if (cov < 1) { const e2 = Math.max(d, 0), gl = (Math.exp(-e2 / 3.2) * .34 + Math.exp(-e2 / 9) * .2) * E.en * (1 - cov); r = r * cov + cb[0] * gl; g = g * cov + cb[1] * gl; b = b * cov + cb[2] * gl; }
        }
        const o = (py * 240 + px) * 3;
        buf[o] = Math.min(255, Math.round(r + buf[o])); buf[o + 1] = Math.min(255, Math.round(g + buf[o + 1])); buf[o + 2] = Math.min(255, Math.round(b + buf[o + 2]));
      }
    }
  }
}

const SEQ = [['idle', 3.4], ['happy', 2.2], ['love', 2.4], ['star', 2.2], ['think', 2.4], ['done', 2.4], ['fail', 2.2], ['scare', 1.6], ['idle', 2.0]];
const out = process.argv[2];
if (!out) { console.error('usage: node ref.mjs OUT.rgb'); process.exit(2); }
const N = Math.round(SEQ.reduce((a, s) => a + s[1], 0) * 60);
const fd = fs.openSync(out, 'w');
let si = 1, nextAt = SEQ[0][1] * 1000;
for (let f = 1; f <= N; f++) {
  const vt = f * 1000 / 60;
  if (si < SEQ.length && vt >= nextAt - 1e-6) { set(SEQ[si][0], vt); nextAt += SEQ[si][1] * 1000; si++; }
  direct(vt, E.key); step(vt, 1 / 60); render(vt);
  fs.writeSync(fd, buf);
}
fs.closeSync(fd);
console.error(`${N} frames -> ${out}`);
