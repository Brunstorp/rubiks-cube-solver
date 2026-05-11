/**
 * Live-scan wizard with manual editing.
 *
 * For each of the 6 faces:
 *   - Live webcam preview on the left, polling /api/detect_face every
 *     ~400 ms; detected sticker boxes are overlaid on the live frame.
 *   - A 3×3 mini-face preview on the right shows the captured state for
 *     the current face. Centre cell is locked to the face letter; the
 *     other 8 cells start empty.
 *   - User can fill cells two ways:
 *       1. Click "Apply camera detection" to copy the latest detection
 *          into the mini face (centre stays locked).
 *       2. Click a palette swatch + click a mini-face cell to paint it
 *          manually. Useful when the cube has text / stickers that
 *          confuse the detector.
 *   - Prev/Next buttons jump between the 6 steps. Next is disabled until
 *     all 9 cells of the current face are filled. The current step's
 *     state is preserved when navigating, so the user can come back to
 *     fix any face.
 *
 * Once all 6 faces are filled, the wizard emits a 54-letter state to the
 * onComplete callback — no /api/classify round-trip needed.
 */

const FACES = ['F', 'U', 'L', 'R', 'D', 'B'];
const PALETTE_ORDER = ['U', 'D', 'F', 'B', 'L', 'R']; // matches main palette

const c = (name) => `<span class="ci ci-${name.toLowerCase()}">${name}</span>`;
const HINT = (text) => `<span class="ci-hint">(${text})</span>`;

const FACE_INSTRUCTIONS = {
    F: `${c('GREEN')} on front, ${c('WHITE')} on top.`,
    U: `${c('WHITE')} on front, ${c('BLUE')} on top. ${HINT('tilt the cube to shoot the top from above')}`,
    L: `${c('ORANGE')} on front, ${c('WHITE')} on top.`,
    R: `${c('RED')} on front, ${c('WHITE')} on top.`,
    D: `${c('YELLOW')} on front, ${c('GREEN')} on top. ${HINT('tilt the cube to shoot the bottom from below')}`,
    B: `${c('BLUE')} on front, ${c('WHITE')} on top.`,
};
const FACE_NAME = {
    F: 'FRONT', U: 'TOP', L: 'LEFT', R: 'RIGHT', D: 'BOTTOM', B: 'BACK',
};

// Overlay rectangle stroke + palette swatch background. Match the main
// palette and the 3D cube so colours line up across the app.
const COLOR_HEX = {
    F: '#1bb24b', U: '#ffffff', L: '#ff7a00', R: '#d80000', D: '#ffd400', B: '#2363e8',
};
const COLOR_TITLE = {
    U: 'White', D: 'Yellow', F: 'Green', B: 'Blue', L: 'Orange', R: 'Red',
};

const POLL_INTERVAL_MS = 400;
const FRAME_MAX_DIM = 640;

const $ = id => document.getElementById(id);

let stream = null;
let currentIdx = 0;
let captured = {};            // face -> 9-array of color letters (null = empty)
let activeColor = 'U';
let onCompleteCallback = null;
let handlersBound = false;
let pollTimer = null;
let pollInflight = false;
let latestDetection = null;

export async function runCaptureWizard(onComplete) {
    onCompleteCallback = onComplete;
    currentIdx = 0;
    latestDetection = null;
    activeColor = 'U';

    // Pre-initialise every face: only the centre is set, the rest is empty.
    captured = {};
    for (const f of FACES) {
        captured[f] = new Array(9).fill(null);
        captured[f][4] = f;
    }

    $('camera-modal').classList.remove('hidden');
    bindHandlersOnce();
    buildMiniPalette();

    $('cam-error').classList.add('hidden');
    await startCamera();
    showFace(currentIdx);
    startPolling();
}

async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError('Camera API not available in this browser. Use the upload link below or paint manually.');
        return;
    }
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: 'environment' } },
            audio: false,
        });
        $('cam-video').srcObject = stream;
    } catch (e) {
        showError(`Camera unavailable: ${e.message}. Paint the face manually or use the upload link below.`);
    }
}

function bindHandlersOnce() {
    if (handlersBound) return;
    handlersBound = true;
    $('cam-prev').onclick = goPrev;
    $('cam-next').onclick = goNext;
    $('cam-apply').onclick = applyDetection;
    $('cam-close').onclick = closeWizard;
    $('cam-file').onchange = handleFileUpload;
}

// ─── Sidebar: mini face + palette ────────────────────────────────────────
function buildMiniPalette() {
    const palette = $('mini-palette');
    palette.innerHTML = '';
    for (const colour of PALETTE_ORDER) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'mini-swatch';
        btn.dataset.color = colour;
        btn.style.background = COLOR_HEX[colour];
        btn.title = `${colour} / ${COLOR_TITLE[colour]}`;
        btn.onclick = () => selectColor(colour);
        palette.appendChild(btn);
    }
    selectColor(activeColor);
}

function selectColor(colour) {
    activeColor = colour;
    $('mini-palette').querySelectorAll('.mini-swatch').forEach(b => {
        b.classList.toggle('active', b.dataset.color === colour);
    });
}

function renderMiniFace() {
    const grid = $('mini-face');
    grid.innerHTML = '';
    const cells = captured[FACES[currentIdx]];
    for (let i = 0; i < 9; i++) {
        const cell = document.createElement('div');
        cell.className = 'mini-cell';
        const v = cells[i];
        if (v) cell.classList.add('filled', `color-${v}`);
        if (i === 4) cell.classList.add('center');
        cell.onclick = () => paintCell(i);
        grid.appendChild(cell);
    }
}

function paintCell(i) {
    if (i === 4) return;  // centre is locked
    captured[FACES[currentIdx]][i] = activeColor;
    renderMiniFace();
    updateFooter();
}

// ─── Step navigation ─────────────────────────────────────────────────────
function showFace(idx) {
    const face = FACES[idx];
    $('cam-title').textContent =
        `Step ${idx + 1} of 6 — ${face} (${FACE_NAME[face]})`;
    $('cam-instruction').innerHTML = FACE_INSTRUCTIONS[face];
    $('cam-progress').innerHTML = FACES.map((f, i) => {
        const cls = i < idx ? 'done' : i === idx ? 'current' : '';
        return `<span class="dot ${cls}">${f}</span>`;
    }).join('');
    $('cam-error').classList.add('hidden');
    latestDetection = null;
    setScanStatus('Scanning…', '');
    clearOverlay();
    renderMiniFace();
    updateFooter();
}

function goPrev() {
    if (currentIdx === 0) return;
    currentIdx--;
    showFace(currentIdx);
}

function goNext() {
    if (!isCurrentFaceComplete()) return;
    if (currentIdx >= FACES.length - 1) {
        finish();
    } else {
        currentIdx++;
        showFace(currentIdx);
    }
}

function isCurrentFaceComplete() {
    return captured[FACES[currentIdx]].every(c => c !== null);
}

function updateFooter() {
    $('cam-prev').disabled = currentIdx === 0;
    $('cam-next').disabled = !isCurrentFaceComplete();
    $('cam-next').textContent = currentIdx >= FACES.length - 1
        ? 'Done ✓'
        : 'Next face →';
    const detected = latestDetection && latestDetection.detected;
    $('cam-apply').disabled = !detected;
}

// ─── Polling loop ────────────────────────────────────────────────────────
function startPolling() {
    stopPolling();
    pollTimer = setInterval(pollOnce, POLL_INTERVAL_MS);
    pollOnce();
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

async function pollOnce() {
    if (pollInflight) return;
    const video = $('cam-video');
    if (!video.srcObject || video.videoWidth === 0) return;

    pollInflight = true;
    try {
        const blob = await videoFrameToBlob(video);
        const form = new FormData();
        form.append('frame', blob, 'frame.jpg');
        const res = await fetch('/api/detect_face', { method: 'POST', body: form });
        if (!res.ok) return;
        const data = await res.json();
        latestDetection = data;
        drawOverlay(data);
        setScanStatus(
            data.detected ? '9 stickers detected ✓' : 'Adjust cube…',
            data.detected ? 'ok' : 'bad',
        );
        updateFooter();
    } catch (_) {
        /* transient errors are silent — next poll retries */
    } finally {
        pollInflight = false;
    }
}

function videoFrameToBlob(video) {
    const w = video.videoWidth, h = video.videoHeight;
    const scale = Math.min(1, FRAME_MAX_DIM / Math.max(w, h));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.85));
}

// ─── Overlay drawing ─────────────────────────────────────────────────────
function drawOverlay(data) {
    const canvas = $('cam-overlay-canvas');
    const display = $('cam-video');
    const w = display.clientWidth, h = display.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    if (!data || !data.detected) return;

    // Map detection-frame coords to displayed video coords, accounting for
    // object-fit: cover cropping.
    const srcW = data.width, srcH = data.height;
    let scale, dx, dy;
    if (srcW / srcH > w / h) {
        scale = h / srcH;
        dx = (w - srcW * scale) / 2;
        dy = 0;
    } else {
        scale = w / srcW;
        dx = 0;
        dy = (h - srcH * scale) / 2;
    }

    for (const s of data.stickers) {
        const x = s.x * scale + dx;
        const y = s.y * scale + dy;
        const sw = s.w * scale;
        const sh = s.h * scale;
        ctx.lineWidth = 3;
        ctx.strokeStyle = COLOR_HEX[s.color] || '#ffffff';
        ctx.shadowColor = 'rgba(0,0,0,0.7)';
        ctx.shadowBlur = 4;
        ctx.strokeRect(x, y, sw, sh);
        ctx.shadowBlur = 0;
    }
}

function clearOverlay() {
    const canvas = $('cam-overlay-canvas');
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

function setScanStatus(msg, cls) {
    const el = $('cam-status');
    el.textContent = msg;
    el.className = `cam-status ${cls}`;
}

// ─── Capture actions ─────────────────────────────────────────────────────
function applyDetection() {
    if (!latestDetection || !latestDetection.detected) return;
    const face = FACES[currentIdx];
    const colors = latestDetection.stickers.map(s => s.color);
    colors[4] = face;  // centre never overwritten
    captured[face] = colors;
    renderMiniFace();
    updateFooter();
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    e.target.value = '';
    if (!file) return;
    setScanStatus('Analysing uploaded image…', '');
    const form = new FormData();
    form.append('frame', file, file.name);
    try {
        const res = await fetch('/api/detect_face', { method: 'POST', body: form });
        const data = await res.json();
        if (!data.detected) {
            showError('No cube face detected in the uploaded image.');
            return;
        }
        const face = FACES[currentIdx];
        const colors = data.stickers.map(s => s.color);
        colors[4] = face;
        captured[face] = colors;
        renderMiniFace();
        updateFooter();
    } catch (err) {
        showError(`Upload failed: ${err.message}`);
    }
}

function finish() {
    // Sanity: every face should be filled by now (Done button was disabled
    // until isCurrentFaceComplete on step 6).
    for (const f of FACES) {
        if (captured[f].some(c => c === null)) {
            showError(`Face ${f} is incomplete.`);
            return;
        }
    }
    const state = [];
    for (const face of FACES) {
        for (const letter of captured[face]) state.push(letter);
    }
    closeWizard();
    onCompleteCallback(state);
}

function closeWizard() {
    stopPolling();
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
    $('cam-video').srcObject = null;
    $('camera-modal').classList.add('hidden');
    clearOverlay();
}

function showError(msg) {
    const el = $('cam-error');
    el.textContent = msg;
    el.classList.remove('hidden');
}
