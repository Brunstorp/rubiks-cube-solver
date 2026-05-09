/**
 * Three.js renderer for a 3x3x3 cube. Maintains a 54-element logical state
 * and rebuilds the visible stickers from it. Supports click-to-paint and
 * animated layer rotations.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const COLORS = {
    U: 0xffffff,  // white
    D: 0xffd400,  // yellow
    F: 0x1bb24b,  // green
    B: 0x1a4dd6,  // blue
    L: 0xff7a00,  // orange
    R: 0xd80000,  // red
};

// Sticker index → cubelet position and face name. Indices match cube.py:
//   F: 0-8, U: 9-17, L: 18-26, R: 27-35, D: 36-44, B: 45-53.
const STICKER_MAP = [
    // F face 0-8 (+Z normal)
    {pos: [-1, +1, +1], face: 'F'}, {pos: [ 0, +1, +1], face: 'F'}, {pos: [+1, +1, +1], face: 'F'},
    {pos: [-1,  0, +1], face: 'F'}, {pos: [ 0,  0, +1], face: 'F'}, {pos: [+1,  0, +1], face: 'F'},
    {pos: [-1, -1, +1], face: 'F'}, {pos: [ 0, -1, +1], face: 'F'}, {pos: [+1, -1, +1], face: 'F'},
    // U face 9-17 (+Y normal)
    {pos: [-1, +1, -1], face: 'U'}, {pos: [ 0, +1, -1], face: 'U'}, {pos: [+1, +1, -1], face: 'U'},
    {pos: [-1, +1,  0], face: 'U'}, {pos: [ 0, +1,  0], face: 'U'}, {pos: [+1, +1,  0], face: 'U'},
    {pos: [-1, +1, +1], face: 'U'}, {pos: [ 0, +1, +1], face: 'U'}, {pos: [+1, +1, +1], face: 'U'},
    // L face 18-26 (-X normal)
    {pos: [-1, +1, -1], face: 'L'}, {pos: [-1, +1,  0], face: 'L'}, {pos: [-1, +1, +1], face: 'L'},
    {pos: [-1,  0, -1], face: 'L'}, {pos: [-1,  0,  0], face: 'L'}, {pos: [-1,  0, +1], face: 'L'},
    {pos: [-1, -1, -1], face: 'L'}, {pos: [-1, -1,  0], face: 'L'}, {pos: [-1, -1, +1], face: 'L'},
    // R face 27-35 (+X normal)
    {pos: [+1, +1, +1], face: 'R'}, {pos: [+1, +1,  0], face: 'R'}, {pos: [+1, +1, -1], face: 'R'},
    {pos: [+1,  0, +1], face: 'R'}, {pos: [+1,  0,  0], face: 'R'}, {pos: [+1,  0, -1], face: 'R'},
    {pos: [+1, -1, +1], face: 'R'}, {pos: [+1, -1,  0], face: 'R'}, {pos: [+1, -1, -1], face: 'R'},
    // D face 36-44 (-Y normal)
    {pos: [-1, -1, +1], face: 'D'}, {pos: [ 0, -1, +1], face: 'D'}, {pos: [+1, -1, +1], face: 'D'},
    {pos: [-1, -1,  0], face: 'D'}, {pos: [ 0, -1,  0], face: 'D'}, {pos: [+1, -1,  0], face: 'D'},
    {pos: [-1, -1, -1], face: 'D'}, {pos: [ 0, -1, -1], face: 'D'}, {pos: [+1, -1, -1], face: 'D'},
    // B face 45-53 (-Z normal)
    {pos: [+1, +1, -1], face: 'B'}, {pos: [ 0, +1, -1], face: 'B'}, {pos: [-1, +1, -1], face: 'B'},
    {pos: [+1,  0, -1], face: 'B'}, {pos: [ 0,  0, -1], face: 'B'}, {pos: [-1,  0, -1], face: 'B'},
    {pos: [+1, -1, -1], face: 'B'}, {pos: [ 0, -1, -1], face: 'B'}, {pos: [-1, -1, -1], face: 'B'},
];

const FACE_NORMAL = {
    F: new THREE.Vector3( 0,  0, +1),
    U: new THREE.Vector3( 0, +1,  0),
    L: new THREE.Vector3(-1,  0,  0),
    R: new THREE.Vector3(+1,  0,  0),
    D: new THREE.Vector3( 0, -1,  0),
    B: new THREE.Vector3( 0,  0, -1),
};

// Move name → axis ('x'/'y'/'z'), which slice (-1, 0, +1), and rotation
// angle in degrees. Sign of the angle matches a real cube turn viewed from
// the corresponding face.
const MOVE_INFO = {
    'U':  {axis: 'y', layer: +1, angle: -90},
    "U'": {axis: 'y', layer: +1, angle: +90},
    'U2': {axis: 'y', layer: +1, angle: 180},
    'D':  {axis: 'y', layer: -1, angle: +90},
    "D'": {axis: 'y', layer: -1, angle: -90},
    'D2': {axis: 'y', layer: -1, angle: 180},
    'R':  {axis: 'x', layer: +1, angle: -90},
    "R'": {axis: 'x', layer: +1, angle: +90},
    'R2': {axis: 'x', layer: +1, angle: 180},
    'L':  {axis: 'x', layer: -1, angle: +90},
    "L'": {axis: 'x', layer: -1, angle: -90},
    'L2': {axis: 'x', layer: -1, angle: 180},
    'F':  {axis: 'z', layer: +1, angle: -90},
    "F'": {axis: 'z', layer: +1, angle: +90},
    'F2': {axis: 'z', layer: +1, angle: 180},
    'B':  {axis: 'z', layer: -1, angle: +90},
    "B'": {axis: 'z', layer: -1, angle: -90},
    'B2': {axis: 'z', layer: -1, angle: 180},
};

export class ThreeCube {
    constructor(canvas, labelContainer) {
        this.canvas = canvas;
        this.labelContainer = labelContainer;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x111114);

        this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
        this.camera.position.set(5, 4, 6.5);

        this.renderer = new THREE.WebGLRenderer({canvas, antialias: true});
        this.renderer.setPixelRatio(window.devicePixelRatio);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.15;
        this.controls.enablePan = false;
        this.controls.minDistance = 5;
        this.controls.maxDistance = 15;

        this.scene.add(new THREE.AmbientLight(0xffffff, 0.85));
        const dir = new THREE.DirectionalLight(0xffffff, 0.5);
        dir.position.set(5, 10, 7);
        this.scene.add(dir);

        this.cubeRoot = new THREE.Group();
        this.scene.add(this.cubeRoot);

        this.state = null;
        this.cubelets = [];
        this.stickers = new Array(54);

        this.editable = true;
        this.stickerClickCallback = null;
        this._isAnimating = false;

        this._setupFaceLabels();
        this._setupRaycaster();
        this._setupResize();

        this._loop = this._loop.bind(this);
        this._loop();
    }

    setState(state54) {
        this.state = [...state54];
        this._rebuild();
    }

    getState() {
        return [...this.state];
    }

    setEditable(editable) {
        this.editable = editable;
    }

    onStickerClick(callback) {
        this.stickerClickCallback = callback;
    }

    paintSticker(idx, color) {
        if (idx < 0 || idx >= 54) return;
        if ((idx % 9) === 4) return;  // never overwrite a centre
        this.state[idx] = color;
        const mesh = this.stickers[idx];
        if (mesh) mesh.material.color.setHex(COLORS[color]);
    }

    /**
     * Animate a single move (e.g. "U2"), then snap to the new logical state.
     * Returns a promise that resolves when the rotation animation completes.
     */
    async animateMoveAndApply(move, newState, durationMs) {
        if (this._isAnimating) return;
        this._isAnimating = true;
        try {
            await this._rotateLayer(move, durationMs);
        } finally {
            // Always rebuild from the authoritative new state so floating-point
            // drift in the rotation can't accumulate.
            this.setState(newState);
            this._isAnimating = false;
        }
    }

    _rotateLayer(move, durationMs) {
        return new Promise((resolve) => {
            const info = MOVE_INFO[move];
            if (!info) { resolve(); return; }

            // Collect all root-level cubeRoot children whose centre lies on
            // the rotating slice. The threshold has to exceed the sticker
            // outward-offset (0.501) so the rotating face's *own* stickers
            // (which sit at layer ± 0.501) get included along with the
            // cubelets (at exactly ± layer) and the wrap-around stickers
            // from neighbouring faces (also at exactly ± layer in this axis).
            const movers = [];
            for (const obj of [...this.cubeRoot.children]) {
                const p = obj.position;
                const v = info.axis === 'x' ? p.x : info.axis === 'y' ? p.y : p.z;
                if (Math.abs(v - info.layer) < 0.6) movers.push(obj);
            }

            const group = new THREE.Group();
            this.scene.add(group);
            for (const o of movers) {
                this.cubeRoot.remove(o);
                group.add(o);
            }

            const axis = info.axis === 'x' ? new THREE.Vector3(1, 0, 0)
                       : info.axis === 'y' ? new THREE.Vector3(0, 1, 0)
                       :                     new THREE.Vector3(0, 0, 1);
            const targetAngle = info.angle * Math.PI / 180;

            const start = performance.now();
            const tick = () => {
                const t = Math.min(1, (performance.now() - start) / durationMs);
                const eased = 1 - Math.pow(1 - t, 3);  // ease-out cubic
                group.setRotationFromAxisAngle(axis, targetAngle * eased);
                if (t < 1) {
                    requestAnimationFrame(tick);
                } else {
                    // Detach the now-rotated objects so the rebuild can replace
                    // them cleanly. The caller's setState() will dispose them.
                    while (group.children.length > 0) {
                        const obj = group.children[0];
                        group.remove(obj);
                        // Reparent back to cubeRoot for the brief window before
                        // setState rebuilds — prevents a 1-frame flash of
                        // missing geometry.
                        this.cubeRoot.add(obj);
                    }
                    this.scene.remove(group);
                    resolve();
                }
            };
            requestAnimationFrame(tick);
        });
    }

    _rebuild() {
        // Tear down existing meshes
        for (const c of this.cubelets) {
            this.cubeRoot.remove(c);
            c.geometry.dispose();
            c.material.dispose();
        }
        for (const s of this.stickers) {
            if (!s) continue;
            this.cubeRoot.remove(s);
            s.geometry.dispose();
            s.material.dispose();
        }
        this.cubelets = [];
        this.stickers = new Array(54);

        const cubeletGeom = new THREE.BoxGeometry(0.96, 0.96, 0.96);
        const cubeletMat = new THREE.MeshLambertMaterial({color: 0x0a0a0c});
        for (let x = -1; x <= 1; x++) {
            for (let y = -1; y <= 1; y++) {
                for (let z = -1; z <= 1; z++) {
                    if (x === 0 && y === 0 && z === 0) continue;
                    const c = new THREE.Mesh(cubeletGeom, cubeletMat);
                    c.position.set(x, y, z);
                    this.cubeRoot.add(c);
                    this.cubelets.push(c);
                }
            }
        }

        const stickerGeom = new THREE.PlaneGeometry(0.86, 0.86);
        for (let idx = 0; idx < 54; idx++) {
            const info = STICKER_MAP[idx];
            const colorKey = this.state[idx];
            const color = COLORS[colorKey] ?? 0x444444;
            const mat = new THREE.MeshBasicMaterial({color});
            const mesh = new THREE.Mesh(stickerGeom, mat);
            const normal = FACE_NORMAL[info.face];
            mesh.position.set(
                info.pos[0] + normal.x * 0.501,
                info.pos[1] + normal.y * 0.501,
                info.pos[2] + normal.z * 0.501,
            );
            mesh.lookAt(
                mesh.position.x + normal.x,
                mesh.position.y + normal.y,
                mesh.position.z + normal.z,
            );
            mesh.userData.stickerIndex = idx;
            this.cubeRoot.add(mesh);
            this.stickers[idx] = mesh;
        }
    }

    _setupFaceLabels() {
        this.faceLabels = {};
        for (const face of ['U', 'D', 'L', 'R', 'F', 'B']) {
            const el = document.createElement('div');
            el.className = 'face-label';
            el.textContent = face;
            this.labelContainer.appendChild(el);
            this.faceLabels[face] = el;
        }
    }

    _updateFaceLabels() {
        const camDir = this.camera.position.clone().normalize();
        const v = new THREE.Vector3();
        for (const face of Object.keys(this.faceLabels)) {
            const normal = FACE_NORMAL[face];
            v.copy(normal).multiplyScalar(1.85);
            v.project(this.camera);
            const x = (v.x * 0.5 + 0.5) * this.canvas.clientWidth;
            const y = (-v.y * 0.5 + 0.5) * this.canvas.clientHeight;
            const el = this.faceLabels[face];
            el.style.left = `${x}px`;
            el.style.top = `${y}px`;
            // Hide labels on faces angled away from the camera.
            el.classList.toggle('hidden', normal.dot(camDir) < 0.1);
        }
    }

    _setupRaycaster() {
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        let downX = 0, downY = 0;
        this.canvas.addEventListener('pointerdown', (e) => {
            downX = e.clientX; downY = e.clientY;
        });
        this.canvas.addEventListener('pointerup', (e) => {
            const dx = e.clientX - downX, dy = e.clientY - downY;
            if (dx*dx + dy*dy > 25) return;  // a drag, not a click
            if (!this.editable || !this.stickerClickCallback || this._isAnimating) return;
            const rect = this.canvas.getBoundingClientRect();
            this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
            this.raycaster.setFromCamera(this.mouse, this.camera);
            const hits = this.raycaster.intersectObjects(this.stickers.filter(Boolean), false);
            if (hits.length > 0) {
                this.stickerClickCallback(hits[0].object.userData.stickerIndex);
            }
        });
    }

    _setupResize() {
        const resize = () => {
            const w = this.canvas.clientWidth;
            const h = this.canvas.clientHeight;
            if (w === 0 || h === 0) return;
            this.renderer.setSize(w, h, false);
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
        };
        window.addEventListener('resize', resize);
        // Defer initial resize so CSS layout has settled.
        requestAnimationFrame(resize);
    }

    _loop() {
        requestAnimationFrame(this._loop);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
        if (this.state) this._updateFaceLabels();
    }
}
