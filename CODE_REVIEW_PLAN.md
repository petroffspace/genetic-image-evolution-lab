# Code Review, Bug Analysis & Optimization Plan

Scope: Entire codebase review across `11,021` lines of Go (`main.go`, `sessions.go`, `preview.go`, `main_test.go`, `render_test.go`, `sessions_test.go`, `preview_test.go`, `bench_test.go`) and `1,924` lines of frontend assets (`static/index.html`, `static/app.js`, `static/styles.css`).

---

## 1. Bugs & Correctness Issues Found

### Bug 1 (High): `NormMode == 2` (Rank Equalization) breaks `timeline` parameter-morph normalization
- **Locations:**
  - `main.go:3885-3946` (`normalizeRegion`)
  - `main.go:6516-6615` (`handleRenderAnimation`)
- **Root Cause:**
  1. At the start of a timeline animation (`main.go:6519-6523`), each cell's first frame is rendered with `g.normCapture = &ranges[i]` and `g.normCaptureQuant = &quants[i]`.
  2. In `normalizeRegion` (`main.go:3885-3931`), when `mode == 2` (rank equalization), the function returns early at line `3930` **without populating `nc.capture` (`ranges[i]`)**, leaving `ranges[i] = [0, 0]`.
  3. During a `params` transition (`main.go:6611-6612`):
     ```go
     interp.normLo = ga.normLo + (gb.normLo-ga.normLo)*e
     interp.normHi = ga.normHi + (gb.normHi-ga.normHi)*e
     ```
     - If cell `a` has `NormMode == 0` or `1` (`ranges[a] = [-2.8, 2.8]`) and cell `b` has `NormMode == 2` (`ranges[b] = [0, 0]`), `interp` uses `ga.NormMode` (`0` or `1`) while its `[normLo, normHi]` range collapses toward `[0, 0]`, crushing the image into a harsh binary threshold before the late crossfade.
     - If both `a` and `b` have `NormMode == 2`, `interp.normQuant` stays frozen at `quants[a]` instead of interpolating between `quants[a]` and `quants[b]`.
- **Fix:** Always populate `nc.capture` (`[min, max]`) in `normalizeRegion` even when `mode == 2`, and when `ga.NormMode == 2 && gb.NormMode == 2` in `params` transitions, interpolate the quantile table `interp.normQuant[k] = quants[a][k]*(1-e) + quants[b][k]*e`.

---

### Bug 2 (High): Reverse-engineering methods (`precise`, `match`, `estimate`) optimize under legacy `RenderVersion = 0` and then display under `RenderVersion = 7`, plus `perturbGenome` violates v7 gene bounds
- **Locations:**
  - `main.go:4501-4571` (`perturbGenome`)
  - `main.go:4736-4750` (`analyzeGenome`)
  - `main.go:5124-5140` (`reverseEngineerEstimate`)
  - `main.go:7475-7494` (`handleUploadImage`)
- **Root Cause:**
  1. `analyzeGenome` and `reverseEngineerEstimate` initialize `Genome{...}` without setting `RenderVersion: renderVersion` (so `RenderVersion == 0`).
  2. Throughout all optimization iterations in `reverseEngineerPrecise`, `reverseEngineerMatch`, and `reverseEngineerEstimate`, candidate genomes are rendered and scored using the **legacy v0 renderer** (8-bit quantized pre-transform field, pre-v1 colorfulness luma, canvas-space sine warp, wide v0 band-limit roll-off).
  3. Once optimization finishes, `handleUploadImage` sets `genome.RenderVersion = renderVersion` (`7`) at `main.go:7493` and renders the cell preview — which can look substantially different from what was scored.
  4. Additionally, `perturbGenome` (`main.go:4505-4566`) mutates `Exponent` down to `0.5`, `ExponentHi` down to `0.5`, `BandLimit` up to `1.0`, `ReliefStrength` up to `2.0`, and `TerraceLevels` up to `15.0` (violating v7 de-noising bounds), never explores `PaletteMode == 2` (OKLCH), and does not call `applyFeatures(&candidate)`.
- **Fix:** Stamp `RenderVersion: renderVersion` in `analyzeGenome` and `reverseEngineerEstimate`, update `perturbGenome` to respect v7 bounds and `applyFeatures`, and include OKLCH (`PaletteMode == 2`) in `perturbGenome`.

---

### Bug 3 (Medium): `synthMatchChannel` breaks Hermitian symmetry when `PhaseJitter > 0`
- **Location:** `main.go:4880-4910` (`synthMatchChannel`)
- **Root Cause:**
  - `synthMatchChannel` loops over the full 2D FFT grid (`y = 0..padSrcH-1`, `x = 0..padSrcW-1`) and adds independent random jitter `(jitRng.Float64()*2 - 1) * cfg.PhaseJitter` to every bin `(x, y)` without mirroring conjugates `data[ym][xm] = conj(data[y][x])`.
  - When `PhaseJitter > 0` (or `PhaseMix < 1` on self-conjugate Nyquist bins), `data` is no longer Hermitian-symmetric, so the inverse FFT produces a complex field and `real(data[y][x])` loses energy to the discarded imaginary part.
  - It also ignores v7's steeper `BandLimit` roll-off (`0.5 * cutoff * cutoff`) and `BreakFreq`/`ExponentHi`.
- **Fix:** Gate on `cfg.RenderVersion >= 7` (preserving v0–v6 match-mode parity) to enforce conjugate symmetry on jittered phases and apply `BreakFreq`/`ExponentHi` and v7 `BandLimit` roll-off.

---

### Bug 4 (Medium): Non-timeline `mode == "morph"` hardcodes `2x` supersampling instead of `supersampleFactor`
- **Location:** `main.go:6676-6679` (`handleRenderAnimation`)
- **Root Cause:**
  - In the timeline `morph` path (`main.go:6578-6579`), keyframe structure fields use `ss := supersampleFactor(genomes[a], fieldTile, req.Width, req.Height)`.
  - In the non-timeline `mode == "morph"` path (`main.go:6676`), `cw, ch := req.Width*2, req.Height*2` is hardcoded. For any canvas with `max(width, height) < 768` (such as SD presets `640×360`, `640×480`, `480×480`, or `256×192` in tests), `renderExactFramed` uses `ss = 3`, so non-timeline `morph` keyframes do not match `crossfade` or `params` keyframes.
- **Fix:** Use `ss := maxInt(supersampleFactor(genomes[seg], fieldTile, req.Width, req.Height), supersampleFactor(genomes[seg+1], fieldTile, req.Width, req.Height))` in the non-timeline `mode == "morph"` branch.

---

### Bug 5 (Medium): `handleEvolve` and `handleGenerateAll` hold `state.mu.Lock()` during multi-second parallel rendering
- **Locations:**
  - `main.go:7114-7160` (`handleEvolve`)
  - `main.go:7165-7185` (`handleGenerateAll`)
- **Root Cause:**
  - Both handlers hold the exclusive write lock `state.mu.Lock()` across the entire `renderGridPreview` / `seedPopulation` execution (~1–4 seconds).
  - While rendering is in progress, every read-only request (`GET /api/grid`, `POST /api/get-genome`, `POST /api/preview-motion`, `GET /api/sessions`) blocks waiting for `state.mu.RLock()`.
- **Fix:** Snapshot `parentGenome`, `donors`, `locked`, and `free` under `state.mu.RLock()`, perform breeding/scouting/rendering outside the lock (matching `handleSessionRestore` in `sessions.go:409-429` and `loadGenomeIntoCell` in `main.go:7405-7411`), and acquire `state.mu.Lock()` only to push history and commit the rendered results.

---

### Bug 6 (Low): Missing HTTP method checks, `handleSaveParams` directory check, and `previewCache` key collision on match-mode uploads
- **Locations:**
  - `main.go:7081-7262`, `main.go:7315-7394`, `main.go:7519-7571`
  - `main.go:7337-7346` (`handleSaveParams`)
  - `static/app.js:177-188` (`fetchPreviewFrames`)
- **Root Cause:**
  - `handleGrid`, `handleEvolve`, `handleGenerateAll`, `handleToggleLock`, `handleSaveImage`, `handleSaveParams`, `handleLoadParams`, `handleUndo`, and `handleGetGenome` do not enforce `r.Method` or cap request body size with `http.MaxBytesReader` (unlike `sessions.go:229-239` `decodeBody`).
  - `handleSaveParams` does not validate `os.Stat(saveDir)` before writing (returning HTTP `500` instead of `400` when `dir` does not exist).
  - `fetchPreviewFrames` keys `previewCache` by `JSON.stringify([gridData[index].genome, motion])`. Because `/api/grid` strips `LumaRef` from `gridData[index].genome`, two uploaded match-mode images with `Seed: 0` can collide in `previewCache` if `LumaW`/`LumaH` and fitted genes match.
- **Fix:** Enforce HTTP methods (`GET`/`POST`) and request size limits consistently, validate `saveDir` in `handleSaveParams`, and include a fingerprint of `gridData[index].image` in the `previewCache` key in `static/app.js`.

---

## 2. Performance Optimizations (Bit-Identical or Numerically Equivalent)

### Perf 1 (High Impact, ~30–40% faster `Generate All` & `Evolve`): Lookup table for `srgbToOKLab` & fixed-array histogram in `measureImage`
- **Location:** `main.go:7617-7675` (`measureImage`)
- **Analysis:**
  - `measureImage` runs 54+ times per `Generate All`.
  - Line `7622` allocates `fine := make(map[int]int)` and performs `12,288` hash-map insertions per scout for keys `int(r>>4)<<8 | int(g>>4)<<4 | int(b>>4)`, which always lie in `[0, 4096)`. Replacing `map[int]int` with a stack array `var fine [4096]int` removes all map allocations and hashing while staying **bit-identical**.
  - Line `7629` calls `srgbToOKLab(float64(r)/255, float64(g)/255, float64(b)/255)`, invoking `math.Pow((c+0.055)/1.055, 2.4)` 3 times per pixel on `uint8` values `0..255`. Precomputing a 256-entry `srgbByteToLinear` lookup table at init eliminates 3 `math.Pow` calls per pixel (**bit-identical**).

### Perf 2 (High Impact, ~25–35% faster `synthChannel` / `synthPair` / `fft2d`): Parallelize spectral bin synthesis & flatten 2D FFT allocations
- **Location:** `main.go:2262-2645` (`synthChannel`, `synthPair`), `main.go:3448-3485` (`warpNoise`)
- **Analysis:**
  - The frequency-domain synthesis loop over `512 × 1024 = 524,288` bins runs **single-threaded**, calling `math.Sqrt`, `math.Log`, `math.Exp`, `math.Atan2`, and `math.Sincos` on one CPU core before `fft2d`.
  - Pre-drawing the sequential RNG phase stream per row (since each row `y` draws a deterministic fixed count of phases) allows the entire frequency-plane synthesis loop to run across all cores via `parallelRows` while producing **100% bit-identical** spectra and images.
  - Allocating one contiguous backing slice `slab := make([]complex128, padW*padH)` and slicing rows from it reduces heap allocations in `synthChannel` / `synthPair` / `warpNoise` from `1,025` allocations to `2`.

### Perf 3 (Medium-High Impact, ~2× faster `colorizeFloat` on large exports): Fold `Colorfulness` into `buildPaletteLUT`
- **Location:** `main.go:3075-3109` (`buildPaletteLUT`) and `main.go:3179-3183` (`colorizeFloat`)
- **Analysis:**
  - In `colorizeFloat`, `Colorfulness` desaturation (`gray := 0.2126*r + 0.7152*g + 0.0722*b; r = gray + (r-gray)*cf`) runs per pixel (`8.3M` pixels on a supersampled FullHD export) after `lut.at(t)`.
  - Because linear LUT interpolation and linear desaturation commute, applying `Colorfulness` once inside `buildPaletteLUT` (on `1,025` LUT entries) eliminates the per-pixel luma calculation and 3 lerps per pixel. (Note: should be checked against tiny rounding differences or gated on `RenderVersion >= 8` if strict v7 byte identity is desired.)

### Perf 4 (Medium Impact, ~3–5× faster optical-flow post-filtering in `computeFlow`): Sorting network in `median3x3` + parallel `blurPlane` / `median3x3`
- **Location:** `main.go:5817-5885` (`blurPlane`, `median3x3`)
- **Analysis:**
  - `median3x3` calls `sort.Float64s(win[:])` at every pixel of every warp iteration, allocating a slice header and running general quicksort on 9 elements.
  - Replacing `sort.Float64s(win[:])` with an inline 19-comparison 9-element median network and parallelizing `median3x3` and `blurPlane` with `parallelRows` eliminates interface overhead and scales across CPU cores (**bit-identical**).

### Perf 5 (Low-Medium Impact): Parallelize `blendImages`
- **Location:** `main.go:5636-5655` (`blendImages`)
- **Analysis:**
  - Parallelizing `blendImages` by row with `parallelRows` speeds up 4K/8K crossfades and parameter-morph endpoint bridges across cores (**bit-identical**).

---

## 3. Implementation Plan

### Phase 1 — Bug Fixes (Correctness & Concurrency)
1. **Fix `NormMode == 2` timeline transitions (`main.go`):**
   - Populate `nc.capture` (`[min, max]`) in `normalizeRegion` even when `mode == 2`.
   - In `handleRenderAnimation` (`timeline` `params` branch), when both `ga.NormMode == 2 && gb.NormMode == 2`, interpolate `interp.normQuant` between `quants[a]` and `quants[b]`.
2. **Align reverse engineering with v7 renderer & gene bounds (`main.go`):**
   - Set `RenderVersion: renderVersion` in `analyzeGenome` and `reverseEngineerEstimate`.
   - Update `perturbGenome` bounds to respect v7 anti-grain limits (`Exponent >= 1.8`, `ExponentHi >= 2.2`, `BandLimit <= 0.65`, `ReliefStrength <= 0.6`, `TerraceLevels <= 12.0`), support `PaletteMode == 2` (OKLCH), and call `applyFeatures(&candidate)`.
3. **Fix `synthMatchChannel` Hermitian symmetry & v7 roll-off (`main.go`):**
   - For `cfg.RenderVersion >= 7`, enforce conjugate symmetry when applying `PhaseJitter` / `PhaseMix`, and honor `BreakFreq`/`ExponentHi` and v7 `BandLimit` roll-off.
4. **Fix non-timeline `morph` supersampling mismatch (`main.go`):**
   - Replace hardcoded `req.Width*2, req.Height*2` in `handleRenderAnimation` (`else if mode == "morph"`) with `supersampleFactor`.
5. **Narrow `state.mu` lock scope (`main.go`):**
   - In `handleEvolve` and `handleGenerateAll`, snapshot state under `RLock()`, execute rendering/seeding without holding any lock, then take `Lock()` only to update cell history, genomes, images, and strengths.
6. **Harden HTTP handlers & preview cache (`main.go`, `static/app.js`):**
   - Enforce HTTP method checks and `http.MaxBytesReader` consistently across all endpoints in `main.go`.
   - Validate `saveDir` exists and is a directory in `handleSaveParams` before writing.
   - Include a fingerprint of `gridData[index].image` in the `previewCache` key in `static/app.js`.

### Phase 2 — Performance Optimizations
1. **`measureImage` fast path (`main.go`):**
   - Replace `map[int]int` with `[4096]int` stack array and precompute 256-entry `srgbByteToLinear` LUT for bit-identical speedup in `seedPopulation`.
2. **Parallelize spectral synthesis & flatten FFT buffers (`main.go`):**
   - Pre-draw row phase buffers sequentially and parallelize the frequency-plane bin loop in `synthChannel` / `synthPair` via `parallelRows`.
   - Use contiguous backing slices for `[][]complex128` FFT grids in `synthChannel`, `synthPair`, and `warpNoise`.
3. **Parallelize `blurPlane`, `median3x3`, and `blendImages` (`main.go`):**
   - Replace `sort.Float64s` in `median3x3` with an inline 9-element median network and parallelize `median3x3`, `blurPlane`, and `blendImages` via `parallelRows`.
4. **Optional v8 consideration (`main.go`):**
   - Folding `Colorfulness` into `buildPaletteLUT` (and/or any behavior-changing optimization) should either verify bit-exactness or bump `renderVersion` to `8` so `v0`–`v7` renders remain byte-exact.

### Phase 3 — Verification & Benchmarking
1. **Add regression tests:**
   - Test `NormMode == 2` timeline `params` transitions (`render_test.go`).
   - Test `analyzeGenome`, `reverseEngineerEstimate`, `reverseEngineerPrecise`, and `reverseEngineerMatch` carry `RenderVersion == renderVersion` and `perturbGenome` respects v7 gene bounds.
   - Test `synthMatchChannel` Hermitian symmetry (zero discarded imaginary residual) with `PhaseJitter > 0`.
   - Test non-timeline `morph` keyframes match `renderExactFramed` dimensions/content at SD sizes (`640×360`).
2. **Run full test suite & benchmarks:**
   - `go test -race ./...` and `go test ./...` (ensuring frozen `v0`–`v6` fixtures and `TestDenoisedPopulationGrainMetrics` pass).
   - Compare before/after timings with `go test -run XXX -bench=. -benchmem ./...`.
3. **Rebuild & restart:**
   - `go build -o lab .` and restart the background web server.
