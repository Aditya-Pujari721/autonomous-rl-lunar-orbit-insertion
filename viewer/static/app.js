import * as T from "./three.js";
const $ = (id) => document.getElementById(id),
  host = $("viewport"),
  scene = new T.Scene();
const renderer = new T.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
host.append(renderer.domElement);
scene.background = new T.Color(0x030407);
const camera = new T.OrthographicCamera(-8, 8, 6, -6, 0.1, 150);
let yaw = 0.12,
  pitch = 0.26,
  zoom = 1;
function frame() {
  const aspect = host.clientWidth / host.clientHeight,
    w = 8.95 / zoom;
  camera.left = -w;
  camera.right = w;
  camera.top = w / aspect;
  camera.bottom = -w / aspect;
  camera.updateProjectionMatrix();
  camera.position.set(
    30 * Math.sin(yaw) * Math.cos(pitch),
    30 * Math.sin(pitch),
    30 * Math.cos(yaw) * Math.cos(pitch),
  );
  const target =
    mode === "training" ? V(0, 0, 0) : moonFrame ? moonHome : earthHome;
  camera.position.add(target);
  camera.lookAt(target);
}
new ResizeObserver(() => {
  renderer.setSize(host.clientWidth, host.clientHeight);
  frame();
}).observe(host);
let drag;
host.onpointerdown = (e) => {
  drag = [e.clientX, e.clientY];
  host.setPointerCapture(e.pointerId);
};
host.onpointermove = (e) => {
  if (drag) {
    yaw -= (e.clientX - drag[0]) * ((0.007 * 2) / 3);
    pitch = T.MathUtils.clamp(
      pitch + (e.clientY - drag[1]) * ((0.007 * 2) / 3),
      -1.48,
      1.48,
    );
    drag = [e.clientX, e.clientY];
    frame();
  }
};
host.onpointerup = host.onpointercancel = () => (drag = null);
host.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    zoom = T.MathUtils.clamp(zoom * Math.exp(-0.001 * e.deltaY), 0.5, 1.5);
    frame();
  },
  { passive: false },
);
scene.add(new T.AmbientLight(0xffffff, 1.1));
const sun = new T.DirectionalLight(0xffffff, 2.7);
sun.position.set(-5, 6, 8);
scene.add(sun);
const loader = new T.TextureLoader();
function globe(r, x, file) {
  const map = loader.load(file);
  map.colorSpace = T.SRGBColorSpace;
  const m = new T.Mesh(
    new T.SphereGeometry(r, 72, 48),
    new T.MeshStandardMaterial({ map, roughness: 1 }),
  );
  m.position.x = x;
  scene.add(m);
  return m;
}
const earth = globe(1.65, -3.6, "earth.jpg"),
  moon = globe(0.825, 3.6, "moon.jpg");
earth.rotation.y = 2.1;
function geometry(points) {
  return new T.BufferGeometry().setFromPoints(points);
}
const V = (x, y, z) => new T.Vector3(x, y, z);
const orbit = new T.LineLoop(
  geometry(
    Array.from({ length: 256 }, (_, i) =>
      V(
        -3.6 + 7.2 * Math.cos((i * Math.PI) / 128),
        0,
        7.2 * Math.sin((i * Math.PI) / 128),
      ),
    ),
  ),
  new T.LineBasicMaterial({ color: 0x556681 }),
);
scene.add(orbit);
const stars = [];
const starColors = [];
for (let i = 0; i < 12000; i++) {
  const z = Math.random() * 2 - 1,
    a = Math.random() * 2 * Math.PI,
    r = 55 + 20 * Math.random();
  const tint = (238 + Math.floor(Math.random() * 18)) / 255;
  starColors.push(
    ...new T.Color().setRGB(tint, tint, 1, T.SRGBColorSpace).toArray(),
  );
  stars.push(
    V(
      r * Math.sqrt(1 - z * z) * Math.cos(a),
      r * z,
      r * Math.sqrt(1 - z * z) * Math.sin(a),
    ),
  );
}
scene.add(
  new T.Points(
    geometry(stars).setAttribute(
      "color",
      new T.Float32BufferAttribute(starColors, 3),
    ),
    new T.PointsMaterial({
      vertexColors: true,
      size: 1.65,
      sizeAttenuation: false,
    }),
  ),
);
const craft = new T.Group();
craft.scale.setScalar(0.75);
scene.add(craft);
function part(shape, color, y) {
  const m = new T.Mesh(
    shape,
    new T.MeshStandardMaterial({ color, metalness: 0.45, roughness: 0.45 }),
  );
  m.position.y = y;
  craft.add(m);
  return m;
}
part(new T.CylinderGeometry(0.12, 0.12, 0.28, 20), 0xa7afbc, 0);
part(new T.ConeGeometry(0.137, 0.2, 24), 0xf4f4ee, 0.24);
part(new T.CylinderGeometry(0.048, 0.09, 0.11, 12), 0x515866, -0.195);
for (let i = 0; i < 4; i++) {
  let b = part(new T.BoxGeometry(0.09, 0.065, 0.065), 0xd5dae2, 0.055);
  b.position.x = 0.125 * Math.cos((i * Math.PI) / 2);
  b.position.z = 0.125 * Math.sin((i * Math.PI) / 2);
}
const flame = part(new T.ConeGeometry(0.075, 0.24, 12), 0xfc3d21, -0.365);
flame.rotation.z = Math.PI;
flame.material = new T.MeshBasicMaterial({ color: 0xfc3d21 });
const arrow = new T.ArrowHelper(
  V(0, 1, 0),
  V(0, 0, 0),
  0.85,
  0xfc3d21,
  0.18,
  0.12,
);
scene.add(arrow);
arrow.visible = flame.visible = false;
function display(x, y, z) {
  const r = Math.hypot(x, y, z),
    s = (0.825 * (r <= 1 ? r : 1 + 1.5 * Math.log(r))) / Math.max(r, 1e-12);
  return V(3.6 + x * s, z * s, -y * s);
}
function rotate(x, y, z, t, f) {
  const a = f.phase,
    c = Math.cos(data.inclination),
    s = Math.sin(data.inclination),
    p = y * c + z * s;
  return [
    x * Math.cos(a) + p * Math.sin(a),
    -x * Math.sin(a) + p * Math.cos(a),
    z * c - y * s,
  ];
}
let data,
  mode = "random",
  elapsed = 0,
  speed = 1,
  playing = true,
  serial = 0,
  lines = [],
  training,
  lastCount = 0;
let moonFrame = false,
  markers = [];
const earthHome = V(-3.6, 0, 0),
  moonHome = V(3.6, 0, 0);
function offset(t) {
  const a = data.omega * t;
  return V(7.2 * (Math.cos(a) - 1), 0, -7.2 * Math.sin(a));
}
function bodies(t) {
  const delta = offset(t);
  earth.position.copy(earthHome);
  moon.position.copy(moonHome);
  if (moonFrame) earth.position.sub(delta);
  else moon.position.add(delta);
  orbit.position.copy(earth.position).sub(earthHome);
  earth.rotation.order = "ZYX";
  earth.rotation.z = 0.409;
  earth.rotation.y = 2.1 + (2 * Math.PI * t) / 86164;
  moon.rotation.y = data.omega * t;
}
function disposePaths() {
  for (const { line } of lines) {
    scene.remove(line);
    line.geometry.dispose();
    line.material.dispose();
  }
  for (const m of markers) {
    scene.remove(m.arrow);
    m.arrow.dispose();
  }
  lines = [];
  markers = [];
}
function buildPaths() {
  disposePaths();
  lines = [trackLine(data.flight, false), trackLine(data.coast, true)];
  for (const b of data.flight.burns) {
    const pos = drawLine(lines[0], b[0]),
      [x, y, z] = rotate(...b.slice(2, 5), b[0], data.flight);
    const dir = V(x, z, -y).normalize(),
      a = new T.ArrowHelper(dir, pos, 0.48, 0xfc3d21, 0.12, 0.06);
    scene.add(a);
    a.visible = false;
    markers.push({ arrow: a, time: b[0], dir });
  }
}
$("centerSwitch").onchange = (e) => {
  moonFrame = e.target.checked;
  if (mode === "flight") buildPaths();
  frame();
};
function trackLine(track, dotted) {
  let points = [],
    times = [],
    radii = [],
    fuel = [],
    speeds = [];
  const rows = track.points;
  for (let i = 0; i < rows.length; i++) {
    let a = rows[i],
      b = rows[i + 1],
      dt = b ? b[0] - a[0] : 0,
      n = Math.max(1, Math.ceil(dt / 100));
    for (let k = 0; k < n; k++) {
      let u = k / n,
        t = a[0] + dt * u,
        xyz = a
          .slice(1, 4)
          .map((v, j) =>
            b
              ? (2 * u ** 3 - 3 * u * u + 1) * v +
                (u ** 3 - 2 * u * u + u) * dt * a[4 + j] +
                (-2 * u ** 3 + 3 * u * u) * b[1 + j] +
                (u ** 3 - u * u) * dt * b[4 + j]
              : v,
          );
      const velocity = a
        .slice(4, 7)
        .map((v, j) =>
          b
            ? ((6 * u * u - 6 * u) * (a[1 + j] - b[1 + j])) / dt +
              (3 * u * u - 4 * u + 1) * v +
              (3 * u * u - 2 * u) * b[4 + j]
            : v,
        );
      speeds.push(1737400 * Math.hypot(...velocity));
      fuel.push(b ? T.MathUtils.lerp(a[7], b[7], u) : a[7]);
      times.push(t);
      radii.push(Math.hypot(...xyz));
      const p = display(...rotate(...xyz, t, track));
      points.push(moonFrame ? p : p.add(offset(t)));
    }
  }
  const line = new T.Line(
    geometry(points),
    dotted
      ? new T.LineDashedMaterial({
          color: 0x5490ff,
          dashSize: 0.08,
          gapSize: 0.08,
        })
      : new T.LineBasicMaterial({ color: 0xfc3d21 }),
  );
  line.computeLineDistances();
  line.geometry.setDrawRange(0, 0);
  scene.add(line);
  return { line, points, times, radii, speeds, fuel };
}
function clear() {
  disposePaths();
  $("charts").hidden = true;
  if (training) {
    scene.remove(training.mesh);
    training.trail.geometry.dispose();
    for (const {line} of training.tracks) {
      line.geometry.dispose();
      line.material.dispose();
    }
    training = null;
  }
  craft.visible = arrow.visible = flame.visible = false;
}
function select(id) {
  for (const key of ["random", "custom", "training"])
    $(key).classList.toggle("on", key === id);
  $("setup").hidden = true;
}
async function flight(parameters) {
  const id = ++serial;
  mode = "loading";
  clear();
  select(parameters ? "custom" : "random");
  $("status").textContent = "Computing approach";
  $("clock").textContent = "";
  try {
    const res = await fetch("/api/flights", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(parameters ? { parameters } : {}),
    });
    const result = await res.json();
    if (id !== serial) return;
    if (!res.ok) throw Error(result.error);
    data = result;
    buildPaths();
    mode = "flight";
    frame();
    $("frames").hidden = false;
    $("frames").style.display = "flex";
    elapsed = 0;
    playing = true;
    craft.visible = true;
    $("status").textContent = "Flight " + result.seed;
    $("key").hidden = false;
    updateButton();
  } catch (e) {
    if (id === serial) $("status").textContent = e.message;
  }
}
const specs = [
  ["radius", "Distance, lunar radii", 10, 14, 12],
  ["speed", "Excess speed, m/s", 400, 1000, 700],
  ["side", "Sideways speed fraction", 0.2, 0.45, 0.32],
  ["fuel", "Propellant, %", 50, 100, 75],
  ["latitude", "Approach latitude, °", -90, 90, 15],
  ["longitude", "Approach longitude, °", -180, 180, 25],
  ["heading", "Heading east of north, °", -180, 180, 30],
  ["phase", "Lunar phase, °", -180, 180, 0],
];
for (const [key, title, min, max, value] of specs) {
  const l = document.createElement("label");
  l.textContent = title;
  let i = document.createElement("input");
  Object.assign(i, {
    name: key,
    type: "number",
    step: "any",
    value,
    required: true,
    title: min + " to " + max,
  });
  l.append(i);
  $("inputs").append(l);
}
$("setup").onsubmit = (e) => {
  e.preventDefault();
  const p = {};
  for (const [key, , lo, hi] of specs) {
    let i = $("setup").elements[key];
    p[key] = T.MathUtils.clamp(i.valueAsNumber, lo, hi);
    i.value = p[key];
  }
  flight(p);
};
$("custom").onclick = () => {
  $("setup").hidden = false;
  $("inputs").querySelector("input").focus();
};
$("close").onclick = () => ($("setup").hidden = true);
$("random").onclick = () => flight();
let trainingPromise;
async function montage() {
  const res = await fetch("training-v2.bin.gz");
  if (!res.ok) throw Error("Training data unavailable");
  const view = new DataView(await new Response(
    res.body.pipeThrough(new DecompressionStream("gzip"))
  ).arrayBuffer());
  const flights = [], totals = [0];
  for (let offset = 0; offset < view.byteLength;) {
    const stage = view.getUint8(offset), count = view.getUint8(offset + 1);
    totals.push(totals.at(-1) + view.getUint8(offset + 2));
    offset += 3;
    flights.push({stage, count, offset});
    offset += count * 16;
  }
  const response = await fetch("training-orbits.json");
  if (!response.ok) throw Error("Capture orbits unavailable");
  return {view, flights, totals, orbits: await response.json()};
}

function capturePosition(orbit, seconds) {
        const [radius, e, initialMean, p, q] = orbit;
        const mean = (initialMean + seconds*Math.PI/5) % (2*Math.PI);
        let anomaly = mean;
        for (let i=0; i<12; i++) {
          anomaly -= (anomaly-e*Math.sin(anomaly)-mean)/(1-e*Math.cos(anomaly));
        }
        const x = radius*(Math.cos(anomaly)-e);
        const y = radius*Math.sqrt(1-e*e)*Math.sin(anomaly);
        return display(...p.map((v,i)=>v*x+q[i]*y));
}

function sampleTraining(recording) {
  const group = new T.Group(), tracks = [];
  const {view, flights, totals} = recording;
  for (let i = 0; i < 100; i++) {
    const lo = Math.floor(i * flights.length / 100);
    const hi = Math.floor((i + 1) * flights.length / 100);
    const captures = Object.keys(recording.orbits);
    const episode = i === 99
      ? Number(captures[Math.floor(Math.random() * captures.length)])
      : lo + Math.floor(Math.random() * (hi - lo));
    const {stage, count, offset} = flights[episode], points = [], times = [];
    for (let j = 0; j < count; j++) {
      const at = offset + 16 * j;
      const value = k => view.getFloat32(at + 4*k, true);
      times.push(value(0));
      points.push(display(value(1),value(2),value(3)));
    }
    const line = new T.Line(geometry(points), new T.LineBasicMaterial({
      color: 0xfc3d21, transparent: true, depthWrite: false
    }));
    group.add(line);
    tracks.push({stage, episode, line, points, times});
  }
  scene.add(group);
  const orbit = recording.orbits[tracks.at(-1).episode];
  const samples = Array.from({length: 361}, (_, i) => capturePosition(orbit, 10*i/360));
  const trail = new T.Line(geometry(samples), tracks.at(-1).line.material);
  trail.geometry.setDrawRange(0, 0);
  group.add(trail);
  return {mesh: group, tracks, totals, orbit, samples, trail};
}

function trainingGraphs(n) {
  const totals = training.totals, total = totals.length - 1;
  for (const [id, rate] of [["captures",false],["captureRate",true]]) {
    const canvas = $(id), ctx = canvas.getContext("2d");
    const value = i => rate ? (totals[i]-totals[Math.max(0,i-128)]) / Math.max(1,Math.min(i,128)) : totals[i];
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);
    ctx.fillStyle = "white"; ctx.font = "20px Georgia";
    ctx.fillText(rate ? "Success rate (last 128)   "+Math.round(100*value(n))+"%" : "Training Successes   "+totals[n],4,23);
    ctx.strokeStyle = "#666"; ctx.beginPath(); ctx.moveTo(5,32);ctx.lineTo(5,h-5);ctx.lineTo(w-5,h-5);ctx.stroke();
    ctx.strokeStyle = "#eee";ctx.lineWidth=2;ctx.beginPath();
    const point = i => [5+(w-10)*i/total,h-5-(h-40)*value(i)/(rate?1:Math.max(1,totals[total]))];
    ctx.moveTo(...point(0));
    for(let i=32;i<n;i+=32) ctx.lineTo(...point(i));
    ctx.lineTo(...point(n));ctx.stroke();
  }
}
$("training").onclick = async () => {
  let id = ++serial;
  clear();
  select("training");
  mode = "loading";
  $("status").textContent = "Loading training flights";
  try {
    const recording = await (trainingPromise ??= montage());
    if (id !== serial) return;
    training = sampleTraining(recording);
    $("charts").hidden = false;
    mode = "training";
    frame();
    earth.position.copy(earthHome);
    moon.position.copy(moonHome);
    orbit.position.set(0, 0, 0);
    $("frames").hidden = true;
    $("frames").style.display = "none";
    elapsed = lastCount = 0;
    playing = true;
    training.mesh.visible = craft.visible = true;
    $("key").hidden = true;
    updateButton();
  } catch (e) {
    trainingPromise = null;
    if (id === serial) $("status").textContent = e.message;
  }
};
function updateButton() {
  $("symbol").textContent = playing ? "Ⅱ" : "▷";
  $("dial").setAttribute("aria-pressed", String(playing));
}
function durationSeconds() {
  return mode === "flight"
    ? (20 * data.flight.points.at(-1)[0]) / data.flight.decision_end
    : 20;
}
function toggle() {
  if (mode === "flight" && elapsed >= durationSeconds()) {
    elapsed = lastCount = 0;
  }
  playing = !playing;
  updateButton();
}
function setSpeed(s) {
  speed = Math.round(T.MathUtils.clamp(s, 0.5, 2) * 10) / 10;
  $("rate").textContent = speed.toFixed(1) + "×";
  $("dial").querySelector("i").style.transform =
    "rotate(" + (speed - 1) * 100 + "deg)";
}
let knob;
function angle(e) {
  const b = $("dial").getBoundingClientRect();
  return Math.atan2(
    e.clientY - b.y - b.height / 2,
    e.clientX - b.x - b.width / 2,
  );
}
$("dial").onpointerdown = (e) => {
  knob = { a: angle(e), s: speed, turn: 0, moved: false };
  $("dial").setPointerCapture(e.pointerId);
};
$("dial").onpointermove = (e) => {
  if (!knob) return;
  let a = angle(e),
    d = a - knob.a;
  knob.turn += Math.atan2(Math.sin(d), Math.cos(d));
  knob.a = a;
  if (Math.abs(knob.turn) > 0.09) knob.moved = true;
  if (knob.moved) setSpeed(knob.s + knob.turn / 2);
};
$("dial").onpointerup = () => {
  if (knob && !knob.moved) toggle();
  knob = null;
};
$("dial").onpointercancel = () => (knob = null);
$("dial").onclick = (e) => {
  if (e.detail === 0) toggle();
};
$("dial").onkeydown = (e) => {
  if (e.key.startsWith("Arrow")) {
    e.preventDefault();
    setSpeed(speed + (["ArrowUp", "ArrowRight"].includes(e.key) ? 0.1 : -0.1));
  }
};
function drawLine({ line, points, times }, t) {
  let l = 0,
    r = times.length;
  while (l < r) {
    let m = (l + r) >> 1;
    if (times[m] <= t) l = m + 1;
    else r = m;
  }
  line.geometry.setDrawRange(0, l);
  let i = Math.max(0, l - 1);
  return points[i]
    .clone()
    .lerp(
      points[Math.min(l, points.length - 1)],
      l < times.length ? (t - times[i]) / (times[l] - times[i]) : 0,
    );
}
let previous = performance.now();
function render(now) {
  requestAnimationFrame(render);
  const dt = (now - previous) / 1000;
  previous = now;
  if (mode === "flight" || mode === "training") {
    const duration = durationSeconds();
    if (playing) elapsed = Math.min(mode === "training" ? Infinity : duration, elapsed + dt * speed);
    if (mode === "flight") {
      const f = data.flight,
        t = Math.min(f.points.at(-1)[0], (elapsed * f.decision_end) / 20);
      bodies(t);
      craft.position.copy(drawLine(lines[0], t));
      drawLine(lines[1], t);
      const k = f.burns.findLastIndex((b) => b[0] <= t),
        burn = f.burns[k];
      const initial = f.points[0],
        v = rotate(...initial.slice(4, 7), 0, f);
      const dir = k >= 0 ? markers[k].dir : V(v[0], v[2], -v[1]).normalize();
      craft.quaternion.setFromUnitVectors(V(0, 1, 0), dir);
      arrow.visible = flame.visible =
        !!burn && t - burn[0] < Math.max(burn[1], f.decision_end * 0.025);
      arrow.position.copy(craft.position);
      arrow.setDirection(dir);
      camera.updateMatrixWorld();
      for (const marker of markers) marker.arrow.visible = marker.time <= t;
      const tr = lines[0],
        idx = tr.line.geometry.drawRange.count - 1,
        j = Math.min(idx + 1, tr.times.length - 1),
        u = j === idx ? 0 : (t - tr.times[idx]) / (tr.times[j] - tr.times[idx]);
      const km = 1737.4 * T.MathUtils.lerp(tr.radii[idx], tr.radii[j], u);
      const interpolate = (values) =>
        T.MathUtils.lerp(values[idx], values[j], u);
      const stats = [
        (t / 3600).toFixed(1) + " h",
        Math.round(km).toLocaleString() + " km from Moon center",
        Math.round(interpolate(tr.speeds)).toLocaleString() + " m/s",
        interpolate(tr.fuel).toFixed(1) + "% fuel",
        (burn?.[6] ?? 0) + " burns",
      ];
      const clock = $("clock");
      if (clock.children.length !== stats.length)
        clock.replaceChildren(
          ...stats.map(() => document.createElement("span")),
        );
      stats.forEach((value, i) => {
        if (clock.children[i].textContent !== value)
          clock.children[i].textContent = value;
      });

      $("status").textContent =
        "Flight " +
        data.seed +
        (t >= f.decision_end
          ? f.capture
            ? " - Captured"
            : f.event
              ? " - Impact"
              : " - No capture"
          : "");
    } else {
      const cursor = Math.min(100, elapsed * 5), k = Math.min(99, Math.floor(cursor));
      const track = training.tracks[k];
      training.tracks.forEach(({line,points}, i) => {
        line.visible = i <= k;
        if (i < k) line.geometry.setDrawRange(0,points.length);
        line.material.opacity = i === k ? 1 : 0.1 + 0.25*Math.exp(-2*(cursor-i-1));
      });
      craft.position.copy(drawLine(track,T.MathUtils.lerp(track.times[0],track.times.at(-1),cursor-k)));
      if (elapsed >= duration) {
        const seconds = elapsed - duration;
        craft.position.copy(capturePosition(training.orbit, seconds));
        const segments = training.trail.geometry.attributes.position.count - 1;
        const phase = Math.min(segments, seconds * segments / 10);
        const index = Math.floor(phase);
        const positions = training.trail.geometry.attributes.position;
        // restore the previous moving endpoint before revealing more of the orbit.
        if (training.tip !== undefined) positions.setXYZ(training.tip, ...training.samples[training.tip].toArray());
        training.tip = index < segments ? index + 1 : undefined;
        if (training.tip !== undefined) positions.setXYZ(training.tip, ...craft.position.toArray());
        positions.needsUpdate = true;
        training.trail.geometry.setDrawRange(0, Math.min(segments + 1, index + 2));
      }
      craft.quaternion.identity();
      const n = Math.round(cursor/100*(training.totals.length-1));
      trainingGraphs(n);
      $("status").textContent = "Stage "+track.stage+" - Flight "+(track.episode+1).toLocaleString() + (elapsed >= duration ? " - Capture orbit" : "");
      $("clock").textContent = "Training history   "+n.toLocaleString()+" - "+(training.totals.length-1).toLocaleString();

    }
    if (mode === "flight" && elapsed === duration && playing) {
      playing = false;
      updateButton();
    }
  }
  renderer.render(scene, camera);
}
flight();
requestAnimationFrame(render);
