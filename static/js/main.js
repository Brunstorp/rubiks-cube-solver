/**
 * App controller: wires the 3D cube, the colour palette, and the solver API
 * together. Holds the current solution and step index, plus a play loop.
 */
import { ThreeCube } from './cube3d.js';
import { runCaptureWizard } from './capture.js';

const SOLVED = 'F'.repeat(9) + 'U'.repeat(9) + 'L'.repeat(9)
             + 'R'.repeat(9) + 'D'.repeat(9) + 'B'.repeat(9);
const SOLVED_ARR = SOLVED.split('');

const cube = new ThreeCube(
    document.getElementById('cube-canvas'),
    document.getElementById('face-labels'),
);
cube.setState(SOLVED_ARR);

let activeColor = 'U';
let solution = null;        // {moves, states, currentStep}
let playTimer = null;

const STEP_DURATION_MS = 320;
const PLAY_GAP_MS = 80;

// ─── Palette ──────────────────────────────────────────────────────────────
function selectColor(color) {
    activeColor = color;
    document.querySelectorAll('.swatch').forEach(b => {
        b.classList.toggle('active', b.dataset.color === color);
    });
}

document.querySelectorAll('.swatch').forEach(btn => {
    btn.addEventListener('click', () => selectColor(btn.dataset.color));
});
selectColor('U');

cube.onStickerClick(idx => {
    cube.paintSticker(idx, activeColor);
    if (solution) clearSolution();
});

// ─── Action buttons ───────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

$('btn-solved').addEventListener('click', () => {
    cube.setState(SOLVED_ARR);
    clearSolution();
    setStatus('Reset to solved.');
});

$('btn-capture').addEventListener('click', () => {
    if (playTimer) togglePlay();
    runCaptureWizard((state) => {
        cube.setState(state);
        clearSolution();
        setStatus('Scanned. Verify the cube and hit Solve.');
    });
});

$('btn-scramble').addEventListener('click', async () => {
    setStatus('Scrambling…');
    try {
        const res = await fetch('/api/scramble', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({}),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'scramble failed');
        cube.setState(data.state);
        clearSolution();
        setStatus(`Scrambled with: ${data.scramble}`);
    } catch (e) { setError(e.message); }
});

$('btn-solve').addEventListener('click', async () => {
    if (playTimer) togglePlay();
    setStatus('Solving…');
    $('btn-solve').disabled = true;
    try {
        const t0 = performance.now();
        const res = await fetch('/api/solve', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({state: cube.getState()}),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'solve failed');
        const dt = (performance.now() - t0).toFixed(0);
        showSolution(data, dt);
    } catch (e) {
        setError(e.message);
    } finally {
        $('btn-solve').disabled = false;
    }
});

// ─── Step controls ────────────────────────────────────────────────────────
$('btn-first').addEventListener('click', () => goToStep(0));
$('btn-prev').addEventListener('click', () => goToStep(solution.currentStep - 1, true));
$('btn-next').addEventListener('click', () => goToStep(solution.currentStep + 1, true));
$('btn-last').addEventListener('click', () => goToStep(solution.states.length - 1));
$('btn-play').addEventListener('click', togglePlay);

// Inverse of a single quarter/half turn: U → U', U' → U, U2 → U2.
function inverseMove(m) {
    if (m.endsWith('2')) return m;
    if (m.endsWith("'")) return m.slice(0, -1);
    return m + "'";
}

// ─── Solution rendering ───────────────────────────────────────────────────
function showSolution(data, dt) {
    solution = {
        moves: data.moves,
        states: data.states,
        currentStep: 0,
    };
    $('solution-panel').classList.remove('hidden');
    $('solution-summary').textContent =
        data.n_moves === 0
            ? 'Already solved.'
            : `${data.n_moves} moves found in ${dt}ms.`;

    const list = $('moves-list');
    list.innerHTML = '';
    data.moves.forEach((m, i) => {
        const chip = document.createElement('span');
        chip.className = 'move-chip';
        chip.textContent = m;
        chip.addEventListener('click', () => goToStep(i + 1));
        list.appendChild(chip);
    });

    cube.setEditable(false);
    cube.setState(data.states[0]);
    updateStepUI();
    setStatus(`Solution found.`);
}

async function goToStep(target, animate = false) {
    if (!solution) return;
    target = Math.max(0, Math.min(solution.states.length - 1, target));
    if (target === solution.currentStep) return;
    if (animate && target === solution.currentStep + 1) {
        // Step forward: animate the move that takes us to `target`.
        const move = solution.moves[solution.currentStep];
        await cube.animateMoveAndApply(move, solution.states[target], STEP_DURATION_MS);
    } else if (animate && target === solution.currentStep - 1) {
        // Step backward: animate the inverse of the move that brought us here.
        // moves[target] is the move that, applied to states[target], yields
        // states[currentStep] — so its inverse undoes that.
        const move = inverseMove(solution.moves[target]);
        await cube.animateMoveAndApply(move, solution.states[target], STEP_DURATION_MS);
    } else {
        cube.setState(solution.states[target]);
    }
    solution.currentStep = target;
    updateStepUI();
}

function togglePlay() {
    if (playTimer) {
        clearTimeout(playTimer);
        playTimer = null;
        $('btn-play').textContent = '▶';
        return;
    }
    if (!solution) return;
    if (solution.currentStep >= solution.states.length - 1) {
        // Restart from the beginning if at the end.
        cube.setState(solution.states[0]);
        solution.currentStep = 0;
        updateStepUI();
    }
    $('btn-play').textContent = '⏸';
    const tick = async () => {
        if (!playTimer || !solution) return;
        if (solution.currentStep >= solution.states.length - 1) {
            playTimer = null;
            $('btn-play').textContent = '▶';
            return;
        }
        await goToStep(solution.currentStep + 1, true);
        if (playTimer) playTimer = setTimeout(tick, PLAY_GAP_MS);
    };
    playTimer = setTimeout(tick, 0);
}

function updateStepUI() {
    if (!solution) return;
    const i = solution.currentStep;
    const total = solution.states.length - 1;
    $('step-indicator').textContent =
        i === 0 ? `Start of solution (${total} moves to go)`
                : i === total ? `Solved! (${total} moves)`
                              : `Step ${i} / ${total} — last move: ${solution.moves[i - 1]}`;

    document.querySelectorAll('.move-chip').forEach((chip, idx) => {
        chip.classList.toggle('played', idx < i - 1);
        chip.classList.toggle('current', idx === i - 1);
    });

    $('btn-first').disabled = i === 0;
    $('btn-prev').disabled = i === 0;
    $('btn-next').disabled = i >= total;
    $('btn-last').disabled = i >= total;
}

function clearSolution() {
    if (playTimer) {
        clearTimeout(playTimer);
        playTimer = null;
        $('btn-play').textContent = '▶';
    }
    solution = null;
    $('solution-panel').classList.add('hidden');
    cube.setEditable(true);
}

// ─── Status bar ───────────────────────────────────────────────────────────
function setStatus(msg) {
    const el = $('status-bar');
    el.textContent = msg;
    el.classList.remove('error');
}

function setError(msg) {
    const el = $('status-bar');
    el.textContent = `Error: ${msg}`;
    el.classList.add('error');
}
