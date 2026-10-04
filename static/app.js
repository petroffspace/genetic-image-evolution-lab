const TOTAL_CELLS = 9;

// Cache of the last fetched grid so saveParams can build the JSON client-side.
let gridData = [];

function fileTimestamp() {
    const d = new Date();
    const p = n => String(n).padStart(2, '0');
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_` +
           `${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

// Saves a blob with the system save dialog (File System Access API).
// Falls back to a regular download on Firefox/Safari.
async function saveBlobLocally(blob, suggestedName) {
    if ('showSaveFilePicker' in window) {
        try {
            const handle = await window.showSaveFilePicker({ suggestedName });
            const writable = await handle.createWritable();
            await writable.write(blob);
            await writable.close();
            return;
        } catch (err) {
            if (err.name === 'AbortError') throw err; // user cancelled — abort
            // any other error: fall through to download fallback
        }
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = suggestedName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    // Revoking synchronously can cancel the download before the browser
    // has started reading the blob (Firefox).
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// POSTs JSON and returns the parsed response, throwing with the server's
// {"error": ...} message on failure.
async function postJSON(url, body) {
    const res = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: body === undefined ? undefined : JSON.stringify(body)
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.error) {
        throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data;
}

async function fetchGrid() {
    showLoading(true);
    try {
        const res = await fetch('/api/grid');
        const data = await res.json();
        renderGrid(data.cells);
    } catch (err) {
        alert('Failed to load grid: ' + err.message);
    } finally {
        showLoading(false);
    }
}

function renderGrid(cells) {
    stopAllPreviews();
    gridGeneration++;
    gridData = cells;
    const grid = document.getElementById('grid');
    grid.innerHTML = '';

    cells.forEach((cell, i) => {
        const cellDiv = document.createElement('div');
        cellDiv.className = 'cell';

        const img = document.createElement('img');
        img.src = cell.image || '';
        img.className = 'cell-img';
        img.onclick = () => evolve(i);

        const btnContainer = document.createElement('div');
        btnContainer.className = 'cell-buttons';

        const lockBtn = document.createElement('button');
        lockBtn.className = cell.locked ? 'btn btn-locked' : 'btn';
        lockBtn.innerText = cell.locked ? '🔒' : '🔓';
        lockBtn.onclick = () => toggleLock(i);

        const loadBtn = document.createElement('button');
        loadBtn.className = 'btn';
        loadBtn.innerText = '📂';
        loadBtn.title = 'Load Parameters';
        loadBtn.onclick = () => loadParams(i);

        const saveParamBtn = document.createElement('button');
        saveParamBtn.className = 'btn';
        saveParamBtn.innerText = '🧬';
        saveParamBtn.title = 'Save Parameters';
        saveParamBtn.onclick = () => saveParams(i);

        const saveImgBtn = document.createElement('button');
        saveImgBtn.className = 'btn';
        saveImgBtn.innerText = '🖼️';
        saveImgBtn.title = 'Save Image';
        saveImgBtn.onclick = () => saveImage(i);

        const undoBtn = document.createElement('button');
        undoBtn.className = 'btn';
        undoBtn.innerText = '↩️';
        undoBtn.title = 'Undo (restore previous genome)';
        undoBtn.onclick = () => undoCell(i);

        const uploadBtn = document.createElement('button');
        uploadBtn.className = 'btn';
        uploadBtn.innerText = '📤';
        uploadBtn.title = 'Upload & Reverse Engineer';
        uploadBtn.onclick = () => uploadImage(i);

        btnContainer.appendChild(lockBtn);
        btnContainer.appendChild(uploadBtn);
        btnContainer.appendChild(loadBtn);
        btnContainer.appendChild(saveParamBtn);
        btnContainer.appendChild(saveImgBtn);
        btnContainer.appendChild(undoBtn);

        const previewBtn = document.createElement('button');
        previewBtn.type = 'button';
        previewBtn.className = 'preview-btn';
        previewBtn.innerText = '▶';
        previewBtn.title = 'Preview live motion (Animation Studio settings)';
        previewBtn.onclick = () => togglePreview(i, img, previewBtn);

        cellDiv.appendChild(img);
        cellDiv.appendChild(previewBtn);
        const badge = strengthBadge(cell.strength);
        if (badge) cellDiv.appendChild(badge);
        cellDiv.appendChild(btnContainer);

        grid.appendChild(cellDiv);
    });
}

// ---- Live motion preview (per cell) ----
// One loop of the cell's live motion (Animation Studio settings), rendered
// by the server at grid size and played at a low frame rate. Playing
// previews re-render when the motion settings change.

const PREVIEW_FPS = 8;
const MAX_PREVIEW_CACHE = 18;
const PREVIEW_REFRESH_DELAY = 350; // ms of quiet before re-rendering
const previewCache = new Map();   // JSON [genome, motion] -> frame data URLs
// cell index -> {img, btn, timer, frames, k, dir}; dir is the cell's
// random drift direction, kept while other settings change.
const previewPlayers = new Map();
// Bumped on every grid render: a preview that finishes loading after the
// grid was rebuilt must not play into the replaced elements.
let gridGeneration = 0;
// Bumped on every settings refresh, so only the latest one is applied.
let previewRefreshToken = 0;
let previewRefreshTimer = null;

function motionIsStill(m) {
    return !(m.flow > 0 || m.drift > 0 || m.color > 0 || m.morph > 0);
}

// The motion a cell previews with: the current settings, with the cell's
// own direction when drift directions are random.
function previewMotionFor(dir) {
    const m = readMotion();
    if (driftRandom() && m.drift > 0) m.drift_dir = dir;
    return m;
}

async function fetchPreviewFrames(index, motion) {
    const key = JSON.stringify([gridData[index].genome, motion]);
    let frames = previewCache.get(key);
    if (!frames) {
        frames = (await postJSON('/api/preview-motion', { index, motion })).frames;
        previewCache.set(key, frames);
        if (previewCache.size > MAX_PREVIEW_CACHE) {
            previewCache.delete(previewCache.keys().next().value); // oldest
        }
    }
    return frames;
}

function stopAllPreviews() {
    previewPlayers.forEach(p => clearInterval(p.timer));
    previewPlayers.clear();
}

function stopPreview(index) {
    const p = previewPlayers.get(index);
    if (!p) return;
    clearInterval(p.timer);
    previewPlayers.delete(index);
    p.img.src = gridData[index].image || '';
    p.btn.innerText = '▶';
    p.btn.classList.remove('playing', 'updating');
}

async function togglePreview(index, img, btn) {
    if (previewPlayers.has(index)) {
        stopPreview(index);
        return;
    }
    const dir = randomDriftDirs(1)[0];
    const motion = previewMotionFor(dir);
    if (motionIsStill(motion)) {
        alert('Live motion is set to Still. Pick a motion in the Animation Studio to preview it.');
        return;
    }
    const gen = gridGeneration;
    btn.disabled = true;
    btn.innerText = '⏳';
    let frames;
    try {
        frames = await fetchPreviewFrames(index, motion);
    } catch (err) {
        if (gen === gridGeneration) {
            btn.disabled = false;
            btn.innerText = '▶';
        }
        alert('Preview failed: ' + err.message);
        return;
    }
    if (gen !== gridGeneration) return;
    btn.disabled = false;

    const p = { img, btn, frames, k: 0, dir };
    img.src = frames[0];
    btn.innerText = '■';
    btn.classList.add('playing');
    p.timer = setInterval(() => {
        p.k = (p.k + 1) % p.frames.length;
        p.img.src = p.frames[p.k];
    }, 1000 / PREVIEW_FPS);
    previewPlayers.set(index, p);
}

function schedulePreviewRefresh() {
    if (!previewPlayers.size) return;
    clearTimeout(previewRefreshTimer);
    previewRefreshTimer = setTimeout(refreshPreviews, PREVIEW_REFRESH_DELAY);
}

// Re-renders every playing preview with the current motion settings. The
// old loop keeps playing (button pulsing) until the new one arrives.
function refreshPreviews() {
    if (motionIsStill(readMotion())) {
        [...previewPlayers.keys()].forEach(stopPreview);
        return;
    }
    const token = ++previewRefreshToken;
    const gen = gridGeneration;
    previewPlayers.forEach(async (p, index) => {
        p.btn.classList.add('updating');
        try {
            const frames = await fetchPreviewFrames(index, previewMotionFor(p.dir));
            if (token !== previewRefreshToken || gen !== gridGeneration || previewPlayers.get(index) !== p) return;
            p.frames = frames;
            p.k %= frames.length;
        } catch (err) {
            console.error('Preview refresh failed:', err);
        }
        if (token === previewRefreshToken) p.btn.classList.remove('updating');
    });
}

// Mutation strength tiers assigned by the server's slotStrengths
// (gentle 0.5 / normal 1.0 / wild 2.5). 0 = not bred by an evolve step.
function strengthBadge(strength) {
    let tier;
    if (!strength) return null;
    if (strength < 1) tier = 'gentle';
    else if (strength > 1) tier = 'wild';
    else tier = 'normal';
    const badge = document.createElement('span');
    badge.className = 'strength-badge strength-' + tier;
    badge.innerText = tier;
    badge.title = `Bred at mutation strength ${strength}×`;
    return badge;
}

async function evolve(index) {
    showLoading(true);
    try {
        await postJSON('/api/evolve', {clicked_index: index});
    } catch (err) {
        showLoading(false);
        alert('Evolve failed: ' + err.message);
        return;
    }
    await fetchGrid();
}

async function generateAll() {
    showLoading(true);
    try {
        await postJSON('/api/generate-all');
    } catch (err) {
        showLoading(false);
        alert('Generate failed: ' + err.message);
        return;
    }
    await fetchGrid();
}

async function toggleLock(index) {
    try {
        await postJSON('/api/toggle-lock', {index});
    } catch (err) {
        alert('Lock toggle failed: ' + err.message);
        return;
    }
    await fetchGrid();
}

async function saveImage(index) {
    const size = prompt('Enter export size (WxH):', '1920x1080');
    if (!size) return;

    showLoading(true);
    try {
        const res = await fetch('/api/render-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index, size })
        });
        if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            throw new Error(data.error || `HTTP ${res.status}`);
        }
        const blob = await res.blob();
        // The server may narrow wide sizes to the preview's 4:3 view, so
        // name the file after the size actually rendered.
        let [w, h] = size.trim().toLowerCase().split('x');
        w = res.headers.get('X-Image-Width') || w;
        h = res.headers.get('X-Image-Height') || h;
        const name = `image_${String(index).padStart(4, '0')}_${w}x${h}_${fileTimestamp()}.png`;
        await saveBlobLocally(blob, name);
    } catch (err) {
        if (err.name !== 'AbortError') alert('Save failed: ' + err.message);
    } finally {
        showLoading(false);
    }
}

async function undoCell(index) {
    try {
        await postJSON('/api/undo', {index});
    } catch (err) {
        alert('Undo failed: ' + err.message);
        return;
    }
    await fetchGrid();
}

async function saveParams(index) {
    // Fetch the FULL genome (luma reference included) on demand — the grid
    // response strips it to keep payloads small.
    showLoading(true);
    let genome;
    try {
        const data = await postJSON('/api/get-genome', {index});
        showLoading(false);
        genome = data.genome;
    } catch (err) {
        showLoading(false);
        alert('Failed to fetch genome: ' + err.message);
        return;
    }
    const saved = {
        version: '1.3',
        cell_index: index,
        timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19),
        genome: genome
    };
    const blob = new Blob([JSON.stringify(saved, null, 2)], { type: 'application/json' });
    const name = `genome_${String(index).padStart(4, '0')}_${fileTimestamp()}.json`;
    try {
        await saveBlobLocally(blob, name);
    } catch (err) {
        if (err.name !== 'AbortError') alert('Save failed: ' + err.message);
    }
}

async function loadParams(index) {
    let file = null;

    // Preferred: native open-file dialog.
    if ('showOpenFilePicker' in window) {
        try {
            const [handle] = await window.showOpenFilePicker({
                types: [{
                    description: 'Genome JSON',
                    accept: { 'application/json': ['.json'] }
                }],
                multiple: false
            });
            file = await handle.getFile();
        } catch (err) {
            if (err.name === 'AbortError') return; // user cancelled
            // other errors: fall through to fallback picker
        }
    }

    // Fallback: classic <input type="file">.
    if (!file) {
        file = await new Promise(resolve => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json,application/json';
            input.onchange = e => resolve(e.target.files[0] || null);
            input.oncancel = () => resolve(null);
            input.click();
        });
    }
    if (!file) return;

    let saved;
    try {
        saved = JSON.parse(await file.text());
    } catch (e) {
        alert('Not a valid JSON file: ' + e.message);
        return;
    }

    showLoading(true);
    try {
        await postJSON('/api/load-params', { index, saved });
        await fetchGrid();
    } catch (err) {
        alert('Request failed: ' + err.message);
    } finally {
        showLoading(false);
    }
}

async function uploadImage(index) {
    // File picker MUST be triggered directly from user gesture (no prompts before it)
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*,.bmp,.png,.jpg,.jpeg,.gif';

    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // NOW ask for method (accepts number or name)
        const methodInput = prompt(
            'Select processing method:\n\n' +
            'Options:\n' +
            '  1 = brute     - Brute Force Search (tries many random genomes)\n' +
            '  2 = hillclimb - Hill Climbing Optimization (refines parameters)\n' +
            '  3 = estimate  - Frequency Analysis + Fine-tune (analyzes FFT)\n' +
            '  4 = precise   - Deterministic parameter fitting (instant, best texture match)\n' +
            '  5 = match     - Phase-preserving match (best visual similarity)\n\n' +
            'Type: 1-5 OR brute, hillclimb, estimate, precise, match',
            '3'
        );
        if (!methodInput) return;

        // Normalize input (trim + lowercase)
        const cleaned = methodInput.trim().toLowerCase();

        // Map number or name to method
        let method = '';
        if (cleaned === '1' || cleaned === 'brute') {
            method = 'brute';
        } else if (cleaned === '2' || cleaned === 'hillclimb') {
            method = 'hillclimb';
        } else if (cleaned === '3' || cleaned === 'estimate') {
            method = 'estimate';
        } else if (cleaned === '4' || cleaned === 'precise') {
            method = 'precise';
        } else if (cleaned === '5' || cleaned === 'match') {
            method = 'match';
        } else {
            alert('Invalid method. Please use 1-5 or brute, hillclimb, estimate, precise, match.');
            return;
        }

        let iterations = 500;
        if (method === 'brute') {
            iterations = parseInt(prompt('Number of candidates to try (100-5000):', '1000')) || 1000;
        } else {
            iterations = parseInt(prompt('Number of optimization iterations (100-3000):', '500')) || 500;
        }

        // Upload and process
        const formData = new FormData();
        formData.append('image', file);
        formData.append('cellIndex', index.toString());
        formData.append('method', method);
        formData.append('iterations', iterations.toString());

        showLoading(true);
        const loadingText = document.getElementById('loading-text');
        if (loadingText) {
            loadingText.innerText =
                `Processing with ${method} (${iterations} iterations)...\nThis may take 10-60 seconds.`;
        }

        try {
            const res = await fetch('/api/upload-image', {
                method: 'POST',
                body: formData
            });
            const data = await res.json().catch(() => ({}));

            showLoading(false);
            if (loadingText) loadingText.innerText = '';

            if (!res.ok || data.error) {
                alert('Error: ' + (data.error || `HTTP ${res.status}`));
            } else {
                await fetchGrid();
                const g = data.genome;
                alert(
                    `Reverse Engineering Complete!\n\n` +
                    `Method: ${data.method}\n` +
                    `Iterations: ${data.iterations}\n` +
                    `Similarity: ${data.similarity.toFixed(1)}%\n\n` +
                    `Genome:\n` +
                    `  Seed: ${g.seed}\n` +
                    `  Exponent: ${g.exponent.toFixed(3)}\n` +
                    `  Band Limit: ${g.band_limit.toFixed(3)}\n` +
                    `  Axis Stretch: ${g.axis_stretch.toFixed(3)}\n` +
                    `  Gamma: ${g.gamma.toFixed(3)}\n` +
                    `  Colorfulness: ${g.colorfulness.toFixed(3)}\n` +
                    `  Mutation Rate: ${g.mutation_rate.toFixed(4)}\n` +
                    `  Mutation Power: ${g.mutation_power.toFixed(2)}\n\n` +
                    `Palette:\n` +
                    `  Pal A: ${g.pal_a.map(v => v.toFixed(2)).join(', ')}\n` +
                    `  Pal B: ${g.pal_b.map(v => v.toFixed(2)).join(', ')}\n` +
                    `  Pal C: ${g.pal_c.map(v => v.toFixed(2)).join(', ')}\n` +
                    `  Pal D: ${g.pal_d.map(v => v.toFixed(2)).join(', ')}`
                );
            }
        } catch (err) {
            showLoading(false);
            if (loadingText) loadingText.innerText = '';
            alert('Request failed: ' + err.message);
        }
    };

    // Trigger file picker immediately (within user gesture)
    input.click();
}

function showLoading(show) {
    let overlay = document.getElementById('loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div style="text-align:center;">
                <div class="spinner"></div>
                <div id="loading-text" style="margin-top:15px; color:#8b5cf6;"></div>
            </div>
        `;
        document.body.appendChild(overlay);
    }
    overlay.style.display = show ? 'flex' : 'none';
    if (!show) {
        document.getElementById('loading-text').innerText = '';
    }
}

// ============================================================================
// RENDERER FEATURE TOGGLES
// ============================================================================

const FEATURE_IDS = ['transform', 'relief', 'spikes', 'chroma', 'cone',
                     'domain_warp', 'symmetry', 'anchor_palette', 'lch_palette', 'layers', 'cellular', 'lic', 'rd'];

async function loadFeatures() {
    try {
        const res = await fetch('/api/features');
        applyFeatureCheckboxes(await res.json());
    } catch (err) {
        console.error('Failed to load feature settings:', err);
    }
}

function applyFeatureCheckboxes(f) {
    FEATURE_IDS.forEach(id => {
        const cb = document.getElementById('feat-' + id);
        if (cb && typeof f[id] === 'boolean') cb.checked = f[id];
    });
}

function setupFeatures() {
    FEATURE_IDS.forEach(id => {
        const cb = document.getElementById('feat-' + id);
        if (!cb) return;
        cb.onchange = async () => {
            // Build the full toggle state from all checkboxes.
            const payload = {};
            FEATURE_IDS.forEach(k => {
                const el = document.getElementById('feat-' + k);
                if (el) payload[k] = el.checked;
            });
            try {
                await fetch('/api/features', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
            } catch (err) {
                alert('Failed to save feature settings: ' + err.message);
            }
        };
    });
}

document.getElementById('generate-btn').onclick = generateAll;
setupFeatures();
loadFeatures();
fetchGrid();

// ============================================================================
// ANIMATION STUDIO
// ============================================================================

// Exact WxH presets per aspect/quality. All values chosen so every side
// stays within the server's 8192-px cap. 21:9 is exactly 7:3; its 8K entry
// is 8064x3456 because a true 7:3 at 8K (10080x4320) would exceed 8192.
const RES_PRESETS = {
    '4:3':  { 'sd': [640, 480],  'hd': [1024, 768],  'fullhd': [1440, 1080], '4k': [2880, 2160],  '8k': [5760, 4320] },
    '16:9': { 'sd': [640, 360],  'hd': [1280, 720],  'fullhd': [1920, 1080], '4k': [3840, 2160],  '8k': [7680, 4320] },
    '21:9': { 'sd': [1120, 480], 'hd': [1680, 720],  'fullhd': [2520, 1080], '4k': [5040, 2160],  '8k': [8064, 3456] },
    '1:1':  { 'sd': [480, 480],  'hd': [720, 720],   'fullhd': [1080, 1080], '4k': [2160, 2160],  '8k': [4320, 4320] },
    '9:16': { 'sd': [360, 640],  'hd': [720, 1280],  'fullhd': [1080, 1920], '4k': [2160, 3840],  '8k': [4320, 7680] },
};

// Ordered keyframe cells for the animation. Two cells is the default; up
// to MAX_ANIM_SOURCES unique cells (the whole grid) can be chained, each
// appearing once.
const MAX_ANIM_SOURCES = TOTAL_CELLS;
let animSources = [0, 1];

function sourceLabel(pos, count) {
    const letter = String.fromCharCode(65 + pos);
    if (count === 1) return 'Source (loop):';
    if (pos === 0) return `Source ${letter} (Start):`;
    if (pos === count - 1) return `Source ${letter} (End):`;
    return `Source ${letter}:`;
}

function renderAnimSources() {
    const container = document.getElementById('anim-sources');
    container.innerHTML = '';
    const count = animSources.length;

    animSources.forEach((cell, pos) => {
        const row = document.createElement('div');
        row.className = 'form-row';

        const label = document.createElement('label');
        label.innerText = sourceLabel(pos, count);

        // Cells already used by other rows are disabled, so every cell
        // appears in the sequence at most once.
        const select = document.createElement('select');
        for (let c = 0; c < TOTAL_CELLS; c++) {
            const opt = document.createElement('option');
            opt.value = c;
            opt.innerText = `Cell ${c}`;
            opt.disabled = c !== cell && animSources.includes(c);
            select.appendChild(opt);
        }
        select.value = cell;
        select.onchange = () => {
            animSources[pos] = parseInt(select.value);
            renderAnimSources();
        };

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'btn btn-remove-source';
        removeBtn.innerText = '✕';
        removeBtn.title = 'Remove this source cell';
        if (count > 1) {
            removeBtn.onclick = () => {
                animSources.splice(pos, 1);
                renderAnimSources();
            };
        } else {
            removeBtn.classList.add('placeholder');
        }

        row.appendChild(label);
        row.appendChild(select);
        row.appendChild(removeBtn);
        container.appendChild(row);
    });

    document.getElementById('add-source-btn').disabled = count >= MAX_ANIM_SOURCES;
    document.getElementById('anim-sources-info').innerText = count === 1
        ? '1 cell · single-cell loop'
        : `${count}/${MAX_ANIM_SOURCES} cells · ${count - 1} transition${count > 2 ? 's' : ''}`;
    updateAnimTotal();
}

// ---- Timeline (frames per cell / per transition) and live motion ----

// Frame count the server will render for the current settings (mirrors
// handleRenderAnimation): holds plus transitions; without holds,
// transitions share their keyframes and the clip ends on the last cell.
function animTotalFrames() {
    const n = animSources.length;
    const hold = Math.max(0, parseInt(document.getElementById('anim-hold').value) || 0);
    const trans = Math.max(0, parseInt(document.getElementById('anim-trans').value) || 0);
    if (hold > 0) return n * hold + (n - 1) * trans;
    return n > 1 ? (n - 1) * trans + 1 : 0;
}

function updateAnimTotal() {
    const info = document.getElementById('anim-total-info');
    if (!info) return;
    const n = animSources.length;
    const hold = parseInt(document.getElementById('anim-hold').value) || 0;
    const fps = parseInt(document.getElementById('anim-fps').value) || 24;
    const total = animTotalFrames();
    if (n === 1 && hold < 2) {
        info.innerText = 'a single cell needs 2+ frames per cell';
        return;
    }
    let text = `total ${total} frames · ${(total / fps).toFixed(1)} s`;
    if (n === 1) text += ' · loops seamlessly';
    info.innerText = text;
}

const MOTION_PRESETS = {
    still:       { flow: 0,   drift: 0, color: 0, morph: 0 },
    gentle:      { flow: 0.5, drift: 1, color: 0, morph: 0.5 },
    flowing:     { flow: 1.5, drift: 1, color: 1, morph: 1 },
    psychedelic: { flow: 2.5, drift: 2, color: 3, morph: 2 },
};

function applyMotionPreset(name) {
    const p = MOTION_PRESETS[name];
    if (!p) return;
    document.getElementById('motion-flow').value = p.flow;
    document.getElementById('motion-drift').value = p.drift;
    document.getElementById('motion-color').value = p.color;
    document.getElementById('motion-morph').value = p.morph;
    updateMotionLabels();
    schedulePreviewRefresh();
}

function updateMotionLabels() {
    ['flow', 'morph'].forEach(k => {
        document.getElementById(`motion-${k}-val`).innerText =
            parseFloat(document.getElementById(`motion-${k}`).value).toString();
    });
}

function readMotion() {
    return {
        flow: parseFloat(document.getElementById('motion-flow').value) || 0,
        drift: parseInt(document.getElementById('motion-drift').value) || 0,
        drift_dir: parseInt(document.getElementById('motion-drift-dir').value) || 0,
        color: parseInt(document.getElementById('motion-color').value) || 0,
        morph: parseFloat(document.getElementById('motion-morph').value) || 0,
    };
}

// ---- Random drift direction per cell ----

function driftRandom() {
    return document.getElementById('motion-drift-random').checked;
}

// One random direction (0..7) per queued cell; consecutive cells always
// differ so every hand-over visibly changes course.
function randomDriftDirs(n) {
    const dirs = [];
    for (let i = 0; i < n; i++) {
        let d;
        do { d = Math.floor(Math.random() * 8); } while (i > 0 && d === dirs[i - 1]);
        dirs.push(d);
    }
    return dirs;
}

function driftDirLabel(d) {
    return document.querySelector(`#motion-drift-dir option[value="${d}"]`).innerText;
}

function updateDriftDirState() {
    document.getElementById('motion-drift-dir').disabled = driftRandom();
}

function addAnimSource() {
    if (animSources.length >= MAX_ANIM_SOURCES) return;
    const free = [...Array(TOTAL_CELLS).keys()].find(c => !animSources.includes(c));
    if (free === undefined) return;
    animSources.push(free);
    renderAnimSources();
}

function updateAnimSizeInfo() {
    const aspect = document.getElementById('anim-aspect').value;
    const quality = document.getElementById('anim-quality').value;
    const preset = RES_PRESETS[aspect] && RES_PRESETS[aspect][quality];
    const info = document.getElementById('anim-size-info');
    if (info) {
        info.innerText = preset ? `${preset[0]}x${preset[1]}` : 'invalid';
    }
}

async function renderAnimation() {
    const hold = Math.max(0, parseInt(document.getElementById('anim-hold').value) || 0);
    const trans = Math.max(0, parseInt(document.getElementById('anim-trans').value) || 0);
    const fps = parseInt(document.getElementById('anim-fps').value);
    const easing = document.getElementById('anim-easing').value;
    const dir = document.getElementById('anim-dir').value;
    const mode = document.getElementById('anim-mode') ?
        document.getElementById('anim-mode').value : 'crossfade';

    // Resolve aspect + quality to an exact WxH preset.
    const aspect = document.getElementById('anim-aspect').value;
    const quality = document.getElementById('anim-quality').value;
    const preset = RES_PRESETS[aspect] && RES_PRESETS[aspect][quality];
    if (!preset) {
        alert('Invalid aspect ratio / quality combination');
        return;
    }
    const width = preset[0];
    const height = preset[1];

    // Show progress
    const progressEl = document.getElementById('render-progress');
    const resultEl = document.getElementById('render-result');
    const statusEl = document.getElementById('render-status');
    const barEl = document.getElementById('render-bar');
    const infoEl = document.getElementById('render-info');

    progressEl.classList.remove('hidden');
    resultEl.classList.add('hidden');
    barEl.value = 0;
    statusEl.innerText = 'Preparing animation...';
    infoEl.innerText = '';

    // Send render request
    const payload = {
        source_cells: animSources.slice(),
        hold_frames: hold,
        transition_frames: trans,
        motion: readMotion(),
        fps: fps,
        width: width,
        height: height,
        easing: easing,
        dir: dir,
        mode: mode
    };
    if (driftRandom() && payload.motion.drift > 0) {
        payload.drift_dirs = randomDriftDirs(payload.source_cells.length);
    }

    try {
        showLoading(true);
        const res = await fetch('/api/render-animation', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        showLoading(false);
        progressEl.classList.add('hidden');

        if (res.ok && data.status === 'rendered') {
            resultEl.className = 'result-message success';
            // textContent, not innerHTML: the paths echo the user-typed
            // directory and must not be parsed as markup.
            resultEl.textContent = `
                ✅ Animation Rendered Successfully!
                Sequence: ${payload.source_cells.map(c => 'Cell ' + c).join(' → ')}${payload.source_cells.length === 1 ? ' (seamless loop)' : ''}
                Timeline: ${hold} frames per cell, ${trans} per transition${payload.drift_dirs ? `
                Drift: ${payload.source_cells.map((c, i) => `Cell ${c} ${driftDirLabel(payload.drift_dirs[i])}`).join(', ')}` : ''}
                Output: ${data.output_dir}
                Total Frames: ${data.total_frames}
                Duration: ${(data.total_frames/fps).toFixed(1)}s @ ${fps}fps
                JSON Metadata: ${data.json_path}

                Next Steps:
                • Import frames into video editor (Premiere, Davinci, etc.)
                • Or use FFmpeg to encode: ffmpeg -framerate ${fps} -i "${data.output_dir}/frame_%05d.png" -c:v libx264 -pix_fmt yuv420p output.mp4
            `;
            console.log('Animation rendered:', data);
        } else {
            throw new Error(data.error || 'Rendering failed');
        }
    } catch (err) {
        showLoading(false);
        progressEl.classList.add('hidden');
        resultEl.className = 'result-message error';
        resultEl.textContent = `❌ Error:\n${err.message}`;
        console.error('Animation error:', err);
    }
}

document.getElementById('render-btn').onclick = renderAnimation;
document.getElementById('add-source-btn').onclick = addAnimSource;
['anim-hold', 'anim-trans', 'anim-fps'].forEach(id => {
    document.getElementById(id).addEventListener('input', updateAnimTotal);
});
['motion-flow', 'motion-morph'].forEach(id =>
    document.getElementById(id).addEventListener('input', updateMotionLabels));
document.querySelectorAll('.motion-presets [data-preset]').forEach(btn => {
    btn.onclick = () => applyMotionPreset(btn.dataset.preset);
});
document.getElementById('motion-drift-random').onchange = updateDriftDirState;
// Playing previews follow the motion settings.
['motion-flow', 'motion-morph'].forEach(id =>
    document.getElementById(id).addEventListener('input', schedulePreviewRefresh));
['motion-drift', 'motion-drift-dir', 'motion-color', 'motion-drift-random'].forEach(id =>
    document.getElementById(id).addEventListener('change', schedulePreviewRefresh));
updateDriftDirState();
updateMotionLabels();
renderAnimSources();
document.getElementById('anim-aspect').onchange = updateAnimSizeInfo;
document.getElementById('anim-quality').onchange = updateAnimSizeInfo;
updateAnimSizeInfo();

// ============================================================================
// SESSIONS
// ============================================================================

// Animation Studio inputs saved with a session, by element id. Values are
// stored as the inputs' strings and only applied when still valid, so
// sessions survive options being added or removed later.
const ANIM_FIELD_IDS = ['anim-hold', 'anim-trans', 'motion-flow', 'motion-morph', 'motion-drift',
    'motion-drift-dir', 'motion-color', 'anim-fps', 'anim-aspect', 'anim-quality',
    'anim-mode', 'anim-easing', 'anim-dir', 'motion-drift-random'];

// Session last saved or restored, highlighted in the list.
let currentSessionId = null;

function readAnimSettings() {
    const fields = {};
    ANIM_FIELD_IDS.forEach(id => {
        const el = document.getElementById(id);
        if (el) fields[id] = el.type === 'checkbox' ? (el.checked ? '1' : '0') : el.value;
    });
    return { source_cells: animSources.slice(), fields };
}

function applyAnimSettings(a) {
    if (!a || typeof a !== 'object') return;
    if (Array.isArray(a.source_cells)) {
        const cells = [];
        a.source_cells.forEach(c => {
            c = parseInt(c);
            if (c >= 0 && c < TOTAL_CELLS && !cells.includes(c)) cells.push(c);
        });
        if (cells.length) animSources = cells.slice(0, MAX_ANIM_SOURCES);
    }
    const fields = a.fields || {};
    ANIM_FIELD_IDS.forEach(id => {
        const el = document.getElementById(id);
        const v = fields[id];
        if (!el || typeof v !== 'string') return;
        if (el.type === 'checkbox') {
            el.checked = v === '1';
            return;
        }
        if (el.tagName === 'SELECT' && ![...el.options].some(o => o.value === v)) return;
        el.value = v;
    });
    updateDriftDirState();
    renderAnimSources();
    updateMotionLabels();
    updateAnimSizeInfo();
}

function setSessionsOpen(open) {
    document.body.classList.toggle('sessions-open', open);
    try { localStorage.setItem('sessionsOpen', open ? '1' : '0'); } catch (e) {}
    if (open) loadSessions();
}

async function loadSessions() {
    const list = document.getElementById('sessions-list');
    try {
        const res = await fetch('/api/sessions');
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
        renderSessions(data.sessions);
    } catch (err) {
        list.innerHTML = '';
        const msg = document.createElement('div');
        msg.className = 'sessions-empty';
        msg.textContent = 'Failed to load sessions: ' + err.message;
        list.appendChild(msg);
    }
}

function renderSessions(sessions) {
    const list = document.getElementById('sessions-list');
    list.innerHTML = '';
    if (!sessions.length) {
        const msg = document.createElement('div');
        msg.className = 'sessions-empty';
        msg.textContent = 'No saved sessions yet. Click “Save current session” to keep this grid.';
        list.appendChild(msg);
        return;
    }
    sessions.forEach(s => list.appendChild(sessionItem(s)));
}

function sessionItem(s) {
    const item = document.createElement('div');
    item.className = 'session-item' + (s.id === currentSessionId ? ' current' : '');

    const thumb = document.createElement('img');
    thumb.className = 'session-thumb';
    thumb.src = s.thumb;
    thumb.alt = s.name;
    thumb.loading = 'lazy';
    thumb.title = 'Restore this session';
    thumb.onclick = () => restoreSession(s);

    const nameRow = document.createElement('div');
    nameRow.className = 'session-name-row';
    const name = document.createElement('span');
    name.className = 'session-name';
    name.textContent = s.name;
    name.title = s.name;
    const editBtn = document.createElement('button');
    editBtn.type = 'button';
    editBtn.className = 'btn btn-icon';
    editBtn.innerText = '✏️';
    editBtn.title = 'Rename';
    editBtn.onclick = () => startRename(s, nameRow);
    nameRow.appendChild(name);
    nameRow.appendChild(editBtn);

    const meta = document.createElement('div');
    meta.className = 'session-meta';
    const created = new Date(s.created);
    meta.textContent = (isNaN(created) ? s.created : created.toLocaleString()) +
        (s.locked ? ` · ${s.locked} locked` : '');

    const actions = document.createElement('div');
    actions.className = 'session-actions';
    const restoreBtn = document.createElement('button');
    restoreBtn.type = 'button';
    restoreBtn.className = 'btn btn-restore';
    restoreBtn.innerText = '↺ Restore';
    restoreBtn.title = 'Restore grid, renderer features and Animation Studio settings';
    restoreBtn.onclick = () => restoreSession(s);
    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.className = 'btn btn-delete';
    deleteBtn.innerText = '🗑️';
    deleteBtn.title = 'Delete session';
    deleteBtn.onclick = () => deleteSession(s);
    actions.appendChild(restoreBtn);
    actions.appendChild(deleteBtn);

    item.appendChild(thumb);
    item.appendChild(nameRow);
    item.appendChild(meta);
    item.appendChild(actions);
    return item;
}

// Swaps the name for an input: Enter or blur saves, Escape cancels.
function startRename(s, nameRow) {
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'session-name-input';
    input.value = s.name;
    input.maxLength = 120;
    nameRow.replaceChildren(input);
    input.focus();
    input.select();

    let done = false;
    const finish = async (save) => {
        if (done) return;
        done = true;
        const name = input.value.trim();
        if (save && name && name !== s.name) {
            try {
                await postJSON('/api/sessions/rename', { id: s.id, name });
            } catch (err) {
                alert('Rename failed: ' + err.message);
            }
        }
        loadSessions();
    };
    input.onkeydown = e => {
        if (e.key === 'Enter') finish(true);
        else if (e.key === 'Escape') finish(false);
    };
    input.onblur = () => finish(true);
}

async function saveSession() {
    const btn = document.getElementById('session-save-btn');
    btn.disabled = true;
    try {
        const data = await postJSON('/api/sessions/save', { animation: readAnimSettings() });
        currentSessionId = data.id;
        if (!document.body.classList.contains('sessions-open')) setSessionsOpen(true);
        else await loadSessions();
    } catch (err) {
        alert('Save session failed: ' + err.message);
    } finally {
        btn.disabled = false;
    }
}

async function restoreSession(s) {
    showLoading(true);
    document.getElementById('loading-text').innerText = `Restoring “${s.name}”...`;
    try {
        const data = await postJSON('/api/sessions/restore', { id: s.id });
        applyFeatureCheckboxes(data.features || {});
        applyAnimSettings(data.animation);
        currentSessionId = s.id;
        await fetchGrid();
        loadSessions();
    } catch (err) {
        showLoading(false);
        alert('Restore failed: ' + err.message);
    }
}

async function deleteSession(s) {
    if (!confirm(`Delete session “${s.name}”? This cannot be undone.`)) return;
    try {
        await postJSON('/api/sessions/delete', { id: s.id });
        if (currentSessionId === s.id) currentSessionId = null;
    } catch (err) {
        alert('Delete failed: ' + err.message);
    }
    loadSessions();
}

document.getElementById('session-save-btn').onclick = saveSession;
document.getElementById('sessions-toggle').onclick = () =>
    setSessionsOpen(!document.body.classList.contains('sessions-open'));
document.getElementById('sessions-close').onclick = () => setSessionsOpen(false);
(() => {
    let open = false;
    try { open = localStorage.getItem('sessionsOpen') === '1'; } catch (e) {}
    if (open) setSessionsOpen(true);
})();
