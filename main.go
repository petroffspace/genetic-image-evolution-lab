package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/gif"
	"image/jpeg"
	"image/png"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	totalCells = 9
	// maxAnimSources caps how many cells one animation sequence may chain:
	// every grid cell, each used once.
	maxAnimSources = totalCells
	imgW           = 256
	imgH           = 192
	Port           = 8989
)

// ============================================================================
// FFT ENGINE
// ============================================================================

// fftPlan caches, per transform size and direction, the bit-reversal
// permutation and every stage's twiddle factors. The twiddles are produced
// by the SAME recurrence (w = 1; w *= wLen) the butterflies used to run
// inline, so planned transforms are bit-identical to the original ones;
// they just stop recomputing it (one complex multiply per butterfly) on
// every row and column of every transform.
type fftPlan struct {
	rev []int          // swap partner per index (only i < rev[i] swaps)
	tw  [][]complex128 // tw[s][j]: twiddle j of stage s (length 2<<s)
}

var fftPlans sync.Map // key: n<<1 | inverse -> *fftPlan

func getFFTPlan(n int, inverse bool) *fftPlan {
	key := n << 1
	if inverse {
		key |= 1
	}
	if p, ok := fftPlans.Load(key); ok {
		return p.(*fftPlan)
	}
	p := &fftPlan{rev: make([]int, n)}
	for i, j := 0, 0; i < n; i++ {
		p.rev[i] = j
		k := n >> 1
		for k > 0 && j >= k {
			j -= k
			k >>= 1
		}
		j += k
	}
	for length := 2; length <= n; length <<= 1 {
		angle := 2.0 * math.Pi / float64(length)
		if inverse {
			angle = -angle
		}
		wLen := complex(math.Cos(angle), math.Sin(angle))
		half := length >> 1
		ws := make([]complex128, half)
		w := complex(1, 0)
		for j := 0; j < half; j++ {
			ws[j] = w
			w *= wLen
		}
		p.tw = append(p.tw, ws)
	}
	actual, _ := fftPlans.LoadOrStore(key, p)
	return actual.(*fftPlan)
}

func fft1d(a []complex128, inverse bool) {
	n := len(a)
	if n <= 1 {
		return
	}
	plan := getFFTPlan(n, inverse)
	for i, j := range plan.rev {
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for s, length := 0, 2; length <= n; s, length = s+1, length<<1 {
		ws := plan.tw[s]
		half := length >> 1
		for i := 0; i < n; i += length {
			lo, hi := a[i:i+half], a[i+half:i+length]
			for j, w := range ws {
				u := lo[j]
				t := w * hi[j]
				lo[j] = u + t
				hi[j] = u - t
			}
		}
	}
	if inverse {
		invN := complex(1.0/float64(n), 0)
		for i := range a {
			a[i] *= invN
		}
	}
}

// fft2d transforms rows and columns, parallelized across CPU cores. Rows
// are independent slices; column workers touch disjoint columns with private
// scratch buffers, so both passes are race-free without locks.
func fft2d(data [][]complex128, inverse bool) {
	rows := len(data)
	cols := len(data[0])

	workers := runtime.NumCPU()
	if workers > rows {
		workers = rows
	}
	if workers > cols {
		workers = cols
	}
	if workers < 1 {
		workers = 1
	}

	var wg sync.WaitGroup

	// Rows.
	rchunk := (rows + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo := w * rchunk
		if lo >= rows {
			break
		}
		hi := lo + rchunk
		if hi > rows {
			hi = rows
		}
		wg.Add(1)
		go func(a, b int) {
			defer wg.Done()
			for r := a; r < b; r++ {
				fft1d(data[r], inverse)
			}
		}(lo, hi)
	}
	wg.Wait()

	// Columns: each worker gets its own scratch buffer.
	cchunk := (cols + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo := w * cchunk
		if lo >= cols {
			break
		}
		hi := lo + cchunk
		if hi > cols {
			hi = cols
		}
		wg.Add(1)
		go func(a, b int) {
			defer wg.Done()
			// Columns are gathered in blocks so each row's cache line is
			// touched once per block instead of once per column; every
			// column still gets the same fft1d on the same data.
			const block = 8
			var bufs [block][]complex128
			for i := range bufs {
				bufs[i] = make([]complex128, rows)
			}
			for c0 := a; c0 < b; c0 += block {
				nb := minInt(block, b-c0)
				for r := 0; r < rows; r++ {
					row := data[r][c0 : c0+nb]
					for i, v := range row {
						bufs[i][r] = v
					}
				}
				for i := 0; i < nb; i++ {
					fft1d(bufs[i], inverse)
				}
				for r := 0; r < rows; r++ {
					row := data[r][c0 : c0+nb]
					for i := range row {
						row[i] = bufs[i][r]
					}
				}
			}
		}(lo, hi)
	}
	wg.Wait()
}

// stage runs fn under a pprof "stage" label. Goroutines started inside
// (parallelRows, fft2d workers) inherit it, so a CPU profile attributes
// every sample to its pipeline stage: go tool pprof -tags shows the split.
// The cost is about a microsecond per call.
func stage(name string, fn func()) {
	pprof.Do(context.Background(), pprof.Labels("stage", name), func(context.Context) { fn() })
}

func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

func freqCoord(idx, size int) float64 {
	if idx <= size/2 {
		return float64(idx)
	}
	return float64(idx - size)
}

func minMax(data []float64) (float64, float64) {
	mn, mx := data[0], data[0]
	for _, v := range data {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	return mn, mx
}

// ============================================================================
// GENOME
// ============================================================================

type Genome struct {
	Seed         int64   `json:"seed"`
	Exponent     float64 `json:"exponent"`
	BandLimit    float64 `json:"band_limit"`
	AxisStretch  float64 `json:"axis_stretch"` // anisotropy: stretches fv before radial freq
	Gamma        float64 `json:"gamma"`        // contrast curve on t before palette
	Colorfulness float64 `json:"colorfulness"` // palette color vs grayscale mix
	// Deprecated: per-pixel salt noise. Every render path (renderClean,
	// animation, reverse engineering) forces both to 0, so new genomes no
	// longer set or evolve them. Kept so old saved genomes still load.
	MutationRate  float64    `json:"mutation_rate"`
	MutationPower float64    `json:"mutation_power"`
	PalA          [3]float64 `json:"pal_a"` // cosine palette: base offset
	PalB          [3]float64 `json:"pal_b"` // amplitude
	PalC          [3]float64 `json:"pal_c"` // frequency
	PalD          [3]float64 `json:"pal_d"` // phase

	// Optional phase-match reference: base64 PNG of the target's normalized
	// luminance. Non-empty => phase-preserving "match" rendering mode.
	LumaRef  string  `json:"luma_ref,omitempty"`
	LumaW    int     `json:"luma_w,omitempty"`
	LumaH    int     `json:"luma_h,omitempty"`
	PhaseMix float64 `json:"phase_mix,omitempty"` // 1.0 = full target phase

	// Match-mode variation: radians of seed-driven random phase jitter.
	// 0 = exact reproduction of the reference layout; higher = wilder.
	PhaseJitter float64 `json:"phase_jitter,omitempty"`

	// Structural genes: geometric framing of the luma reference.
	// Zoom 1 = full frame; Rot in radians; Flip* are 0/1; Center in [-0.5, 0.5].
	Zoom    float64 `json:"zoom,omitempty"`
	Rot     float64 `json:"rot,omitempty"`
	FlipX   float64 `json:"flip_x,omitempty"`
	FlipY   float64 `json:"flip_y,omitempty"`
	CenterX float64 `json:"center_x,omitempty"`
	CenterY float64 `json:"center_y,omitempty"`

	// Morph gene: 1.0 = full imported-image structure, 0.0 = pure random
	// spectral synthesis. Drifts downward over generations.
	Structure float64 `json:"structure,omitempty"`

	// Warp: strength of a seed-driven smooth displacement field applied to
	// the imported structure. 0 = rigid, ~0.3 = strong liquid morphing.
	Warp float64 `json:"warp,omitempty"`

	// --- v2 visual genes (defaults reproduce the classic look) ---

	// Nonlinear transform of the normalized luminance before the palette:
	// 0 = none, 1 = turbulence (1-|2t-1|), 2 = ridged (squared turbulence),
	// 3 = terraces (soft-quantized levels).
	Transform     int     `json:"transform,omitempty"`
	TerraceLevels float64 `json:"terrace_levels,omitempty"`

	// Relief lighting: shade added from the luminance gradient.
	ReliefAngle    float64 `json:"relief_angle,omitempty"`    // light direction (radians)
	ReliefStrength float64 `json:"relief_strength,omitempty"` // 0 = off

	// Spectral breakpoint: ExponentHi replaces Exponent for frequencies
	// above BreakFreq*maxFreq. BreakFreq 0 = single-exponent (classic).
	ExponentHi float64 `json:"exponent_hi,omitempty"`
	BreakFreq  float64 `json:"break_freq,omitempty"`

	// Spectral spikes: a few seed-chosen frequency bins get their amplitude
	// multiplied by SpikeAmp -> quasi-periodic stripes / lattice motifs.
	SpikeCount int     `json:"spike_count,omitempty"`
	SpikeAmp   float64 `json:"spike_amp,omitempty"`

	// Chroma modulation: a second, independent field offsets t before the
	// palette, giving spatially-rich iridescent color variation.
	ChromaStrength float64 `json:"chroma_strength,omitempty"`

	// Spectral rotation rotates the fu/fv axes before the radial frequency
	// computation -> diagonal grain instead of axis-aligned streaks.
	SpecRot float64 `json:"spec_rot,omitempty"`

	// Directional cone: frequencies outside +/-ConeWidth*pi around ConeAngle
	// are damped -> brushed / woven looks. ConeWidth 0 or 1 = disabled.
	ConeAngle float64 `json:"cone_angle,omitempty"`
	ConeWidth float64 `json:"cone_width,omitempty"`

	// Domain warp (classic mode): smooth seed-driven displacement applied to
	// the synthesized field before normalization -> marble / flow looks.
	DomainWarp float64 `json:"domain_warp,omitempty"`

	// Tile-space warp shape (render version 3+). WarpScale is the swirl
	// size: the warp noise's spectral cutoff in cycles per tile (larger =
	// smaller, busier swirls; 0 means the default 3). WarpNest 2 nests the
	// warp IQ-style, p + k*r(p + k*q(p)), folding the flow into itself for
	// true marbling; 1 (or 0) is a single displacement.
	WarpScale float64 `json:"warp_scale,omitempty"`
	WarpNest  int     `json:"warp_nest,omitempty"`

	// Normalization mode: 0 = min-max, 1 = percentile clip, 2 = rank equalize.
	NormMode int `json:"norm_mode,omitempty"`

	// Radial symmetry: SymmetryFold k (>=2) folds the luminance k-fold
	// around the image center; <2 = off. SymmetryMirror mirrors alternate
	// wedges -> kaleidoscope / mandala structures.
	SymmetryFold   int  `json:"sym_fold,omitempty"`
	SymmetryMirror bool `json:"sym_mirror,omitempty"`

	// Palette family: 0 = classic cosine palette (PalA-PalD), 1 = anchor
	// points: AnchorCount (2..5) sRGB stops from AnchorColors, lerped over
	// t (in OKLab from render version 2), 2 = OKLCH cosine palette (Lch).
	PaletteMode  int           `json:"palette_mode,omitempty"`
	AnchorCount  int           `json:"anchor_count,omitempty"`
	AnchorColors [5][3]float64 `json:"anchor_colors,omitempty"`
	Lch          LchPalette    `json:"lch"`

	// Layer is an optional second structure field composited onto the
	// main one (classic mode). Mode 0 = no layer.
	Layer LayerSpec `json:"layer"`

	// Cell is an optional cellular (Worley) base field blended with, or
	// replacing, the spectral one (classic mode). Mode 0 = none.
	Cell CellSpec `json:"cell"`

	// Lic optionally smears white noise along a flow field (line integral
	// convolution) and blends the streaks into the structure: brush
	// strokes, hair, fur. Mode 0 = none.
	Lic LicSpec `json:"lic"`

	// RD optionally grows a Gray-Scott reaction-diffusion pattern out of
	// the structure field. Mode 0 = none.
	RD RDSpec `json:"rd"`

	// Seed blending: every frequency bin's random phase is rotated from
	// Seed's realization toward SeedB's along the shortest arc, weighted
	// by SeedBlend in [0,1]. This makes "change the seed" a continuous
	// gene: a child can sit partway between two noise realizations
	// instead of jumping to an unrelated one. Only the phases blend;
	// spike positions, domain warp and the morph mask stay derived from
	// Seed. Inactive when SeedB == 0 or SeedBlend == 0.
	SeedB     int64   `json:"seed_b,omitempty"`
	SeedBlend float64 `json:"seed_blend,omitempty"`

	// Phase-domain morphing (animation only, never serialized): when
	// phaseTo.seed != 0, synthChannel rotates each frequency bin's phase
	// from this genome's realization toward phaseTo's realization,
	// weighted by phaseBlend in [0,1]. Structure then transforms
	// continuously instead of via opacity crossfade.
	phaseTo    phaseSource `json:"-"`
	phaseBlend float64     `json:"-"`

	// Live motion (animation only, never serialized; see animateGenome).
	// All zero = a still render.
	motionFlow float64 `json:"-"` // flow strength: spectral phases churn
	// motionMorph: shape-shift strength; spectral phases wander along a
	// closed loop (see morphPhase) and the structure tile is standardized
	// so the frozen normalization range stays valid as genes oscillate.
	motionMorph float64 `json:"-"`
	motionTau   float64 `json:"-"` // loop position in [0, 1)
	driftU      float64 `json:"-"` // texture offset in tile widths
	driftV      float64 `json:"-"`
	palShift    float64 `json:"-"` // anchor palette ping-pong shift
	// Frozen normalization (animation only): min-max / percentile modes
	// use [normLo, normHi] instead of the frame's own range, so moving
	// content does not pump the brightness. normCapture, if set, receives
	// the range a render computed (to freeze it for later frames).
	normFix     bool        `json:"-"`
	normLo      float64     `json:"-"`
	normHi      float64     `json:"-"`
	normCapture *[2]float64 `json:"-"`
	// Rank mode's frozen equivalent: a quantile table of the first
	// frame's values (normQuant), captured through normCaptureQuant.
	normQuant        []float64  `json:"-"`
	normCaptureQuant *[]float64 `json:"-"`
	// tileCache (animation only): a cell's tile-space work (warp, fold,
	// chroma tile, composed structure tile) built once and shared by all
	// its frames. Only set when that work cannot change between frames
	// (no live flow), so cached frames are bit-identical to uncached ones.
	tileCache *tileCacheEntry `json:"-"`

	// scoutTile (internal, never serialized) realizes the classic field on
	// a smaller tile than fieldTile for cheap population-seeding scouts
	// (see renderScout). Frequency-relative genes stay measured against
	// fieldTile, so a scout is a band-limited, statistically equivalent
	// stand-in for the real render (a different phase draw, same look).
	scoutTile int `json:"-"`

	// RenderVersion selects renderer behavior so improvements never change
	// how previously saved genomes look. 0 (absent) = legacy pipeline;
	// every genome created by this build is stamped renderVersion.
	RenderVersion int `json:"render_version,omitempty"`
}

// renderVersion is stamped on every newly created genome.
//
//	1: colorfulness desaturates toward the palette color's own luma (not
//	   the raw pre-transform field); domain warp and the match-mode morph
//	   mask are isotropic and centered, so they agree across aspect ratios.
//	2: float pipeline: the normalized field stays float32 through
//	   transform, relief, gamma and palette (no 256-level banding), colors
//	   come from a float palette table, and the output is quantized once
//	   with an ordered dither. Anchor palettes interpolate in OKLab.
//	3: domain warp runs in tile space on periodic spectral noise (see
//	   tileWarp) instead of three canvas-space sines: richer, seamless,
//	   identical at every resolution, and it carries the chroma field too.
//	4: canvases smaller than the synthesis tile are supersampled enough
//	   (supersampleFactor) that the palette sees the field at near tile
//	   resolution; grid previews take this path too. Before, each pixel
//	   averaged up to 5x5 tile samples BEFORE the steep palette, and the
//	   leftover sub-pixel variation came out as color grain.
//	5: radial symmetry folds tile coordinates (tileFold) instead of the
//	   finished canvas, so corners fold onto real periodic texture rather
//	   than clamped, smeared border pixels; the chroma field folds with
//	   the structure; match mode reflects at the canvas edge.
//	6: the luminance and chroma tiles are synthesized together (synthPair):
//	   shared per-bin geometry and ONE complex inverse FFT for both real
//	   fields. Mathematically the same fields; floating-point rounding
//	   differs (~1e-16), hence a version: older genomes stay byte-exact.
const renderVersion = 6

// RDSpec describes Gray-Scott reaction-diffusion on the synthesis tile:
//
//	du/dt = Du*lap(u) - u*v^2 + Feed*(1-u)
//	dv/dt = Dv*lap(v) + u*v^2 - (Feed+Kill)*v
//
// seeded from the structure field itself: where the (standardized) field
// exceeds Threshold, the tile starts with chemical V. Feed and Kill pick
// the pattern family (spots, mitosis, stripes, labyrinths, coral, holes;
// see rdPresets); Steps sets how far the pattern grows out of the seed,
// from an organic distortion of the original shapes to a fully developed
// Turing pattern. The standardized V field blends with the structure by
// Mix. It runs on a fixed 256x256 periodic grid per tile (rdGrid), so it
// is resolution-invariant and seamless; Du=0.2, Dv=0.1, dt=1.
type RDSpec struct {
	Mode      int     `json:"mode,omitempty"`
	Feed      float64 `json:"feed,omitempty"`
	Kill      float64 `json:"kill,omitempty"`
	Steps     int     `json:"steps,omitempty"`
	Threshold float64 `json:"threshold,omitempty"`
	Mix       float64 `json:"mix,omitempty"`
}

// rdPresets are (Feed, Kill) pairs inside Gray-Scott's pattern-forming
// region (Pearson's classification); outside it the tile goes uniform.
var rdPresets = [][2]float64{
	{0.035, 0.065},   // spots
	{0.0367, 0.0649}, // mitosis
	{0.060, 0.062},   // stripes
	{0.037, 0.060},   // fingerprint / labyrinth (Pearson's "worms" 0.078/0.061 dies at these diffusion rates)
	{0.0545, 0.062},  // coral
	{0.039, 0.058},   // holes
}

func randomRD(rng *rand.Rand) RDSpec {
	p := rdPresets[rng.Intn(len(rdPresets))]
	return RDSpec{
		Mode:      1,
		Feed:      p[0] + (rng.Float64()*2-1)*0.002,
		Kill:      p[1] + (rng.Float64()*2-1)*0.001,
		Steps:     int(400 * math.Pow(7.5, rng.Float64())), // 400..3000, log-uniform
		Threshold: 0.3 + rng.Float64()*0.9,
		Mix:       0.6 + rng.Float64()*0.4,
	}
}

// rdGrid is the reaction-diffusion grid side for a tile: 256 for the real
// 1024 tile, 128 for 256-tile scouts (diffusion is rescaled so patterns
// keep their size in tile units; scouts are statistically equivalent).
func rdGrid(tile int) int { return clampi(tile/4, 128, 256) }

// grayScott runs steps of Gray-Scott on periodic m x m fields u, v (in
// place, float32 for speed) with diffusion scaled by dscale.
func grayScott(u, v []float32, m, steps int, feed, kill, dscale float64) {
	du, dv := float32(0.2*dscale), float32(0.1*dscale)
	f, fk := float32(feed), float32(feed+kill)
	u2, v2 := make([]float32, m*m), make([]float32, m*m)
	prev, next := make([]int, m), make([]int, m) // wrapped row neighbors
	for i := 0; i < m; i++ {
		prev[i], next[i] = (i+m-1)%m, (i+1)%m
	}
	// cell updates one cell from its four neighbors' values (same
	// arithmetic for interior and edge columns).
	cell := func(uu, vv, ul, ur, uup, udn, vl, vr, vup, vdn float32) (float32, float32) {
		lu := ul + ur + uup + udn - 4*uu
		lv := vl + vr + vup + vdn - 4*vv
		uvv := uu * vv * vv
		return uu + du*lu - uvv + f*(1-uu), vv + dv*lv + uvv - fk*vv
	}
	for s := 0; s < steps; s++ {
		parallelRows(m, func(ya, yb int) {
			for y := ya; y < yb; y++ {
				// Row slices let the compiler hoist bounds checks out of
				// the interior loop; only the two edge columns wrap.
				uc, uU, uD := u[y*m:y*m+m], u[prev[y]*m:prev[y]*m+m], u[next[y]*m:next[y]*m+m]
				vc, vU, vD := v[y*m:y*m+m], v[prev[y]*m:prev[y]*m+m], v[next[y]*m:next[y]*m+m]
				uo, vo := u2[y*m:y*m+m], v2[y*m:y*m+m]
				uo[0], vo[0] = cell(uc[0], vc[0], uc[m-1], uc[1], uU[0], uD[0], vc[m-1], vc[1], vU[0], vD[0])
				for x := 1; x < m-1; x++ {
					uu, vv := uc[x], vc[x]
					lu := uc[x-1] + uc[x+1] + uU[x] + uD[x] - 4*uu
					lv := vc[x-1] + vc[x+1] + vU[x] + vD[x] - 4*vv
					uvv := uu * vv * vv
					uo[x] = uu + du*lu - uvv + f*(1-uu)
					vo[x] = vv + dv*lv + uvv - fk*vv
				}
				uo[m-1], vo[m-1] = cell(uc[m-1], vc[m-1], uc[m-2], uc[0], uU[m-1], uD[m-1], vc[m-2], vc[0], vU[m-1], vD[m-1])
			}
		})
		u, u2 = u2, u
		v, v2 = v2, v
	}
	if steps%2 == 1 { // results live in the swapped buffers
		copy(u2, u)
		copy(v2, v)
	}
}

// applyRD grows the reaction-diffusion pattern from the n x n structure
// tile base and blends it in (in place).
func applyRD(base []float64, n int, cfg Genome) {
	r := cfg.RD
	m := rdGrid(n)
	k := maxInt(n/m, 1)
	seed := make([]float64, m*m)
	for y := 0; y < m; y++ {
		for x := 0; x < m; x++ {
			var sum float64
			for dy := 0; dy < k; dy++ {
				for dx := 0; dx < k; dx++ {
					sum += base[((y*k+dy)%n)*n+(x*k+dx)%n]
				}
			}
			seed[y*m+x] = sum / float64(k*k)
		}
	}
	standardize(seed)
	u, v := make([]float32, m*m), make([]float32, m*m)
	for i, a := range seed {
		s := smoothstep01((a-r.Threshold)/0.6 + 0.5)
		u[i], v[i] = float32(1-0.5*s), float32(0.25*s)
	}
	scale := float64(m) / 256.0
	grayScott(u, v, m, clampi(r.Steps, 0, 6000), clampF(r.Feed, 0.005, 0.1), clampF(r.Kill, 0.04, 0.075), scale*scale)
	vf := make([]float64, m*m)
	for i := range v {
		vf[i] = float64(v[i])
	}
	out := make([]float64, n*n)
	sc := float64(m) / float64(n)
	parallelRows(n, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			for x := 0; x < n; x++ {
				out[y*n+x] = samplePeriodic(vf, m, (float64(x)+0.5)*sc-0.5, (float64(y)+0.5)*sc-0.5)
			}
		}
	})
	standardize(out)
	standardize(base)
	mix := clampF(r.Mix, 0, 1)
	for i := range base {
		base[i] = base[i]*(1-mix) + out[i]*mix
	}
}

// LicSpec describes line integral convolution on the synthesis tile: white
// noise (hashed from Seed+9300) is averaged along the streamline through
// each point, so it turns into streaks that follow a flow field:
//
//	Mode 1 contour: the flow runs along the isolines of the (smoothed)
//	                structure field, so strokes wrap around its shapes.
//	                Smooth is the gradient blur radius.
//	Mode 2 swirl:   an independent divergence-free flow (the curl of smooth
//	                noise, Seed+9301, FlowScale cycles per tile): hair, fur.
//
// Width sets the stroke width: the noise lives on a (512 >> Width)-cell
// grid per tile side, so strokes are 1, 2, 4 or 8 units wide (0 reads as
// fine hair, 2-3 as brush strokes). Length is the streak half-length and
// Smooth the blur radius, both in units of 1/512 tile, so everything
// means the same at any tile size. The
// standardized streaks blend with the standardized structure by Mix. LIC
// runs at half the tile's resolution (streaks are smooth along the flow)
// and is upsampled, after cellular and layer composition and before warp
// and symmetry, so strokes warp, fold and tile with everything else.
type LicSpec struct {
	Mode      int     `json:"mode,omitempty"`
	Length    float64 `json:"length,omitempty"`
	Mix       float64 `json:"mix,omitempty"`
	FlowScale float64 `json:"flow_scale,omitempty"`
	Smooth    float64 `json:"smooth,omitempty"`
	Width     int     `json:"width,omitempty"`
}

func randomLic(rng *rand.Rand) LicSpec {
	l := LicSpec{
		Mode:      1 + rng.Intn(2),
		Length:    8 + rng.Float64()*24,
		Mix:       0.2 + rng.Float64()*0.4,
		FlowScale: 1.5 + rng.Float64()*4.5,
		Smooth:    8 + rng.Float64()*32,
	}
	switch r := rng.Float64(); {
	case r < 0.3:
		l.Width = 0
	case r < 0.65:
		l.Width = 1
	case r < 0.9:
		l.Width = 2
	default:
		l.Width = 3
	}
	return l
}

// licTile computes the standardized LIC streak texture for the n x n
// periodic structure tile base.
func licTile(base []float64, n int, cfg Genome) []float64 {
	l := cfg.Lic
	m := maxInt(n/2, 32)       // LIC grid side
	unit := float64(m) / 512.0 // LIC pixels per 1/512 tile
	steps := maxInt(1, int(math.Round(clampF(l.Length, 1, 60)*unit)))

	bil := func(f []float64, x, y float64) float64 { return samplePeriodic(f, m, x, y) }

	// Noise: white noise on an nN x nN grid per tile (nN = 512 >> Width:
	// the stroke width, defined in tile units), blended between seeds (and
	// toward an animation partner); LIC output is restandardized afterward.
	nN := 512 >> clampi(l.Width, 0, 3)
	toN := float64(nN) / float64(m) // LIC pixels -> noise cells
	noiseOf := func(p phaseSource) []float64 {
		out := make([]float64, nN*nN)
		for y := 0; y < nN; y++ {
			for x := 0; x < nN; x++ {
				v := cellHash(p.seed, x, y, 0) - 0.5
				if p.blended() {
					v += (cellHash(p.seedB, x, y, 0) - 0.5 - v) * clampF(p.blend, 0, 1)
				}
				out[y*nN+x] = v
			}
		}
		return out
	}
	noiseAt := func(f []float64, x, y float64) float64 { return samplePeriodic(f, nN, x*toN, y*toN) }
	noise := noiseOf(cfg.phases().offset(9300))
	if cfg.phaseTo.seed != 0 {
		to := noiseOf(cfg.phaseTo.offset(9300))
		b := clampF(cfg.phaseBlend, 0, 1)
		for i := range noise {
			noise[i] += (to[i] - noise[i]) * b
		}
	}

	vx, vy := licFlow(base, n, m, cfg)

	// Integrate: Hann-weighted average of noise along the streamline,
	// stepping one LIC pixel at a time in both directions.
	w := make([]float64, steps+1)
	for k := range w {
		w[k] = 0.5 + 0.5*math.Cos(math.Pi*float64(k)/float64(steps+1))
	}
	// flowAt bilinearly samples both flow components with one index
	// computation (the integration loop's hot spot).
	mask := m - 1
	flowAt := func(x, y float64) (float64, float64) {
		fx, fy := math.Floor(x), math.Floor(y)
		ix, iy := int(fx), int(fy)
		tx, ty := x-fx, y-fy
		x0, y0, x1, y1 := ix&mask, iy&mask, (ix+1)&mask, (iy+1)&mask
		a, b, c, d := y0*m+x0, y0*m+x1, y1*m+x0, y1*m+x1
		u := (vx[a]*(1-tx)+vx[b]*tx)*(1-ty) + (vx[c]*(1-tx)+vx[d]*tx)*ty
		v := (vy[a]*(1-tx)+vy[b]*tx)*(1-ty) + (vy[c]*(1-tx)+vy[d]*tx)*ty
		return u, v
	}
	lic := make([]float64, m*m)
	parallelRows(m, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			for x := 0; x < m; x++ {
				sum, wsum := noiseAt(noise, float64(x), float64(y))*w[0], w[0]
				for _, dir := range [2]float64{1, -1} {
					px, py := float64(x), float64(y)
					for k := 1; k <= steps; k++ {
						ux, uy := flowAt(px, py)
						// Plain sqrt: unit-scale vectors need none of
						// math.Hypot's overflow care, which cost 13%.
						if d := math.Sqrt(ux*ux + uy*uy); d > 1e-12 {
							ux, uy = ux/d, uy/d
						}
						px, py = px+dir*ux, py+dir*uy
						sum += noiseAt(noise, px, py) * w[k]
						wsum += w[k]
					}
				}
				lic[y*m+x] = sum / wsum
			}
		}
	})

	// Upsample to the tile and standardize.
	out := make([]float64, n*n)
	s := float64(m) / float64(n)
	parallelRows(n, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			for x := 0; x < n; x++ {
				out[y*n+x] = bil(lic, (float64(x)+0.5)*s-0.5, (float64(y)+0.5)*s-0.5)
			}
		}
	})
	standardize(out)
	return out
}

// licFlow computes the unit flow field on the m x m LIC grid for the
// n x n structure tile base (see LicSpec for the two modes).
func licFlow(base []float64, n, m int, cfg Genome) (vx, vy []float64) {
	l := cfg.Lic
	unit := float64(m) / 512.0
	wrap := func(i int) int { return ((i % m) + m) % m }
	vx, vy = make([]float64, m*m), make([]float64, m*m)
	switch l.Mode {
	case 2: // swirl: curl of smooth noise
		scale := l.FlowScale
		if scale <= 0 {
			scale = 3
		}
		side := warpSideFor(cfg)
		psi := warpNoise(cfg.Seed+9301, scale, side)
		toF := float64(side) / float64(m)
		for y := 0; y < m; y++ {
			for x := 0; x < m; x++ {
				fx, fy := (float64(x)+0.5)*toF, (float64(y)+0.5)*toF
				dx := samplePeriodic(psi, side, fx+toF, fy) - samplePeriodic(psi, side, fx-toF, fy)
				dy := samplePeriodic(psi, side, fx, fy+toF) - samplePeriodic(psi, side, fx, fy-toF)
				vx[y*m+x], vy[y*m+x] = dy, -dx
			}
		}
	default: // contour: along isolines of the smoothed structure
		k := n / m
		b := make([]float64, m*m)
		for y := 0; y < m; y++ {
			for x := 0; x < m; x++ {
				var sum float64
				for dy := 0; dy < k; dy++ {
					for dx := 0; dx < k; dx++ {
						sum += base[(y*k+dy)*n+x*k+dx]
					}
				}
				b[y*m+x] = sum / float64(k*k)
			}
		}
		r := maxInt(1, int(math.Round(clampF(l.Smooth, 1, 40)*unit)))
		for pass := 0; pass < 2; pass++ { // two box passes ~ Gaussian
			b = boxBlurPeriodic(b, m, r)
		}
		for y := 0; y < m; y++ {
			for x := 0; x < m; x++ {
				gx := b[y*m+wrap(x+1)] - b[y*m+wrap(x-1)]
				gy := b[wrap(y+1)*m+x] - b[wrap(y-1)*m+x]
				vx[y*m+x], vy[y*m+x] = -gy, gx
			}
		}
	}
	for i := range vx {
		if d := math.Hypot(vx[i], vy[i]); d > 1e-12 {
			vx[i], vy[i] = vx[i]/d, vy[i]/d
		} else {
			vx[i], vy[i] = 1, 0
		}
	}

	return vx, vy
}

// boxBlurPeriodic is a periodic separable box blur of radius r.
func boxBlurPeriodic(f []float64, n, r int) []float64 {
	tmp, out := make([]float64, n*n), make([]float64, n*n)
	inv := 1 / float64(2*r+1)
	for y := 0; y < n; y++ {
		var s float64
		for k := -r; k <= r; k++ {
			s += f[y*n+((k%n)+n)%n]
		}
		for x := 0; x < n; x++ {
			tmp[y*n+x] = s * inv
			s += f[y*n+(x+r+1)%n] - f[y*n+((x-r)%n+n)%n]
		}
	}
	for x := 0; x < n; x++ {
		var s float64
		for k := -r; k <= r; k++ {
			s += tmp[(((k%n)+n)%n)*n+x]
		}
		for y := 0; y < n; y++ {
			out[y*n+x] = s * inv
			s += tmp[((y+r+1)%n)*n+x] - tmp[(((y-r)%n+n)%n)*n+x]
		}
	}
	return out
}

// applyLic blends LIC streaks into the structure tile (in place).
func applyLic(base []float64, n int, cfg Genome) {
	lic := licTile(base, n, cfg)
	standardize(base)
	mix := clampF(cfg.Lic.Mix, 0, 1)
	for i := range base {
		base[i] = base[i]*(1-mix) + lic[i]*mix
	}
}

// CellSpec describes a cellular (Worley) field on the synthesis tile: one
// feature point per grid cell (Density cells per tile side), placed by a
// hash of Seed+8111 and Jitter (0 = regular lattice, 1 = anywhere in its
// cell). Each tile point measures the Metric distance (0 Euclidean, 1
// Manhattan, 2 Chebyshev) to its nearest points:
//
//	Mode 1 F1:     distance to the nearest point: cells, bubbles, pebbles.
//	Mode 2 F2-F1:  second-nearest minus nearest: a network of cracks along
//	               cell borders.
//	Mode 3 value:  the nearest point's random value: a flat-shaded mosaic
//	               (scales, stained glass).
//
// The field is standardized and blended with the standardized spectral
// field by Mix (1 = cellular only, the spectral FFT is then skipped). It
// lives on the periodic tile like everything else, so it tiles, warps,
// folds and layers seamlessly at any resolution. Seed blending and the
// animation morph move each feature point continuously between seeds.
type CellSpec struct {
	Mode    int     `json:"mode,omitempty"`
	Density float64 `json:"density,omitempty"`
	Jitter  float64 `json:"jitter,omitempty"`
	Metric  int     `json:"metric,omitempty"`
	Mix     float64 `json:"mix,omitempty"`
}

func randomCell(rng *rand.Rand) CellSpec {
	c := CellSpec{
		Mode:    1 + rng.Intn(3),
		Density: 4 * math.Pow(6, rng.Float64()), // 4..24 cells per side, log-uniform
		Jitter:  0.6 + rng.Float64()*0.4,
		Mix:     0.5 + rng.Float64()*0.5,
	}
	if rng.Float64() < 0.2 {
		c.Jitter = rng.Float64() * 0.6 // near-regular lattices: tiles, honeycombs
	}
	if r := rng.Float64(); r < 0.15 {
		c.Metric = 1
	} else if r < 0.3 {
		c.Metric = 2
	}
	if rng.Float64() < 0.4 {
		c.Mix = 1
	}
	return c
}

// cellCount is the integer number of cells per tile side (integer, so the
// lattice tiles seamlessly).
func (c CellSpec) cellCount() int { return maxInt(1, int(math.Round(c.Density))) }

// cellHash maps (seed, ix, iy, k) to a uniform float in [0, 1)
// (splitmix64): feature points are a pure function of their cell, so any
// tile size and any canvas see the same points.
func cellHash(seed int64, ix, iy, k int) float64 {
	z := uint64(seed) ^ uint64(ix)*0x9E3779B97F4A7C15 ^ uint64(iy)*0xC2B2AE3D27D4EB4F ^ uint64(k)*0x165667B19E3779F9
	z += 0x9E3779B97F4A7C15
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	z ^= z >> 31
	return float64(z>>11) / (1 << 53)
}

// cellPoints realizes the feature points (cell-local x, y in [0, 1)) and
// values of an n x n lattice for a phase source, blending each point from
// Seed's realization toward SeedB's by the blend amount.
func cellPoints(p phaseSource, n int, jitter float64) [][3]float64 {
	pts := make([][3]float64, n*n)
	at := func(seed int64, ix, iy int) [3]float64 {
		return [3]float64{
			0.5 + jitter*(cellHash(seed, ix, iy, 0)-0.5),
			0.5 + jitter*(cellHash(seed, ix, iy, 1)-0.5),
			cellHash(seed, ix, iy, 2),
		}
	}
	for iy := 0; iy < n; iy++ {
		for ix := 0; ix < n; ix++ {
			q := at(p.seed, ix, iy)
			if p.blended() {
				r := at(p.seedB, ix, iy)
				b := clampF(p.blend, 0, 1)
				for k := range q {
					q[k] += (r[k] - q[k]) * b
				}
			}
			pts[iy*n+ix] = q
		}
	}
	return pts
}

// worleyTile computes the cellular field on an n x n periodic tile.
func worleyTile(n int, cfg Genome) []float64 {
	c := cfg.Cell
	cells := c.cellCount()
	jitter := clampF(c.Jitter, 0, 1)
	src := cfg.phases().offset(8111)
	pts := cellPoints(src, cells, jitter)
	if cfg.phaseTo.seed != 0 { // animation: move points toward the partner's
		to := cellPoints(cfg.phaseTo.offset(8111), cells, jitter)
		b := clampF(cfg.phaseBlend, 0, 1)
		for i := range pts {
			for k := range pts[i] {
				pts[i][k] += (to[i][k] - pts[i][k]) * b
			}
		}
	}
	if cfg.motionFlow > 0 {
		// Live flow: every point swings to a second realization and back a
		// whole number of times per loop. Blending two points of the same
		// jitter box stays inside it, so the pruned neighbor search holds.
		alt := cellPoints(src.offset(1), cells, jitter)
		k := math.Max(1, math.Round(cfg.motionFlow))
		w := (1 - math.Cos(2*math.Pi*k*cfg.motionTau)) / 2
		for i := range pts {
			for j := range pts[i] {
				pts[i][j] += (alt[i][j] - pts[i][j]) * w
			}
		}
	}
	if cfg.motionMorph > 0 {
		// Shape-shift: points lean partway toward another realization and
		// back once per loop (staying inside their jitter box, as above).
		alt := cellPoints(src.offset(2), cells, jitter)
		w := math.Min(1, 0.2*cfg.motionMorph) * (1 - math.Cos(2*math.Pi*cfg.motionTau)) / 2
		for i := range pts {
			for j := range pts[i] {
				pts[i][j] += (alt[i][j] - pts[i][j]) * w
			}
		}
	}
	out := make([]float64, n*n)
	scale := float64(cells) / float64(n) // tile pixels -> cell units
	parallelRows(n, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			for x := 0; x < n; x++ {
				out[y*n+x] = worleyAt((float64(x)+0.5)*scale, (float64(y)+0.5)*scale, cells, pts, c.Metric, c.Mode, jitter)
			}
		}
	})
	return out
}

// worleyOrder lists the 5x5 neighborhood offsets nearest-first (own cell,
// the 8 around it, then the outer 16), so pruning cuts in early.
var worleyOrder = func() [][2]int {
	var o [][2]int
	for ring := 0; ring <= 2; ring++ {
		for dy := -ring; dy <= ring; dy++ {
			for dx := -ring; dx <= ring; dx++ {
				if maxInt(absInt(dx), absInt(dy)) == ring {
					o = append(o, [2]int{dx, dy})
				}
			}
		}
	}
	return o
}()

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// worleyAt evaluates the cellular field at (fx, fy) in cell units on a
// periodic cells x cells lattice, for points jittered by jitter. A 5x5
// neighborhood: with full jitter the second-nearest point (F2), and under
// Manhattan even the nearest, can lie two cells away. Cells are visited
// nearest-first and skipped when the closest a point inside their jitter
// box could be cannot beat the current F1 (modes 1, 3) or F2 (mode 2),
// which is exact. Euclidean distances compare squared.
func worleyAt(fx, fy float64, cells int, pts [][3]float64, metric, mode int, jitter float64) float64 {
	cx, cy := int(math.Floor(fx)), int(math.Floor(fy))
	lo, hi := 0.5-jitter/2, 0.5+jitter/2 // cell-local range of a point
	// dist uses plain comparisons, not math.Max: inputs are never NaN, and
	// math.Max's NaN/signed-zero handling was a quarter of the render.
	dist := func(dx, dy float64) float64 {
		switch metric {
		case 1:
			return dx + dy
		case 2:
			if dx > dy {
				return dx
			}
			return dy
		}
		return dx*dx + dy*dy // squared; rooted at the end
	}
	f1, f2, val := math.Inf(1), math.Inf(1), 0.0
	for _, o := range worleyOrder {
		ix, iy := cx+o[0], cy+o[1]
		bound := f1
		if mode == 2 {
			bound = f2
		}
		// Gap from the query to the cell's jitter box, per axis.
		gx, gy := 0.0, 0.0
		if d := float64(ix) + lo - fx; d > 0 {
			gx = d
		} else if d := fx - float64(ix) - hi; d > 0 {
			gx = d
		}
		if d := float64(iy) + lo - fy; d > 0 {
			gy = d
		} else if d := fy - float64(iy) - hi; d > 0 {
			gy = d
		}
		if dist(gx, gy) >= bound {
			continue
		}
		wx, wy := ix%cells, iy%cells
		if wx < 0 {
			wx += cells
		}
		if wy < 0 {
			wy += cells
		}
		p := pts[wy*cells+wx]
		dx, dy := float64(ix)+p[0]-fx, float64(iy)+p[1]-fy
		if dx < 0 {
			dx = -dx
		}
		if dy < 0 {
			dy = -dy
		}
		d := dist(dx, dy)
		if d < f1 {
			f1, f2, val = d, f1, p[2]
		} else if d < f2 {
			f2 = d
		}
	}
	if metric == 0 {
		f1, f2 = math.Sqrt(f1), math.Sqrt(f2)
	}
	switch mode {
	case 2:
		return f2 - f1
	case 3:
		return val
	}
	return f1
}

// LayerSpec describes a second spectral field B, realized on the same
// tile from Seed+7001 with its own spectral genes, and how it combines
// with the main field A. Both are standardized (zero mean, unit std) on
// the tile first, so neither dominates by raw amplitude. Composition
// happens in tile space before resampling, so it inherits resolution
// independence, seamless tiling, the domain warp and supersampling.
//
//	Mode 1 mask:     a smooth mask field m (Seed+7100, MaskScale cycles per
//	                 tile) picks B where sigmoid(MaskSharp*(m+MaskBias)) is
//	                 high: patches of one texture inside another.
//	Mode 2 modulate: f = A * (1 + Mix*tanh(B)): B sets A's local contrast,
//	                 giving calm and turbulent regions. The envelope is
//	                 bounded; an exp(B) envelope's tail blew up the range
//	                 and normalization flattened the rest of the image.
//	Mode 3 max:      f = lerp(A, max(A, B+2*MaskBias), Mix): B overlays A
//	                 wherever it rises above it, like stacked strata.
//
// Mix scales how strongly B enters (mask and max: blend amount).
type LayerSpec struct {
	Mode        int     `json:"mode,omitempty"`
	Exponent    float64 `json:"exponent,omitempty"`
	ExponentHi  float64 `json:"exponent_hi,omitempty"`
	BreakFreq   float64 `json:"break_freq,omitempty"`
	BandLimit   float64 `json:"band_limit,omitempty"`
	AxisStretch float64 `json:"axis_stretch,omitempty"`
	SpecRot     float64 `json:"spec_rot,omitempty"`
	ConeAngle   float64 `json:"cone_angle,omitempty"`
	ConeWidth   float64 `json:"cone_width,omitempty"`
	Mix         float64 `json:"mix,omitempty"`
	MaskScale   float64 `json:"mask_scale,omitempty"`
	MaskSharp   float64 `json:"mask_sharp,omitempty"`
	MaskBias    float64 `json:"mask_bias,omitempty"`
}

// randomLayer draws a layer with independent spectral genes (so B
// usually contrasts with A) in one of the three modes.
func randomLayer(rng *rand.Rand) LayerSpec {
	l := LayerSpec{
		Mode:        1 + rng.Intn(3),
		Exponent:    1.5 + rng.Float64()*2.0,
		ExponentHi:  1.5 + rng.Float64()*2.0,
		BreakFreq:   0.15 + rng.Float64()*0.5,
		BandLimit:   0.3 + rng.Float64()*0.6,
		AxisStretch: 0.5 + rng.Float64()*1.5,
		SpecRot:     rng.Float64() * 2 * math.Pi,
		ConeAngle:   rng.Float64() * 2 * math.Pi,
		ConeWidth:   1.0,
		Mix:         0.4 + rng.Float64()*0.6,
		MaskScale:   1.5 + rng.Float64()*3.5,
		MaskSharp:   1.5 * math.Pow(16, rng.Float64()), // 1.5..24, log-uniform
		MaskBias:    rng.Float64() - 0.5,
	}
	if rng.Float64() < 0.4 {
		l.ConeWidth = 0.2 + rng.Float64()*0.6
	}
	return l
}

// layerGenome is the genome that realizes layer B: the main genome with
// B's spectral genes, no spikes, and every phase seed (own, seed blend,
// animation partner) offset by 7001 so B blends and morphs between the
// matching B realizations, as the chroma field does with 9091.
func layerGenome(cfg Genome) Genome {
	l := cfg.Layer
	g := cfg
	g.Exponent, g.ExponentHi, g.BreakFreq = l.Exponent, l.ExponentHi, l.BreakFreq
	g.BandLimit, g.AxisStretch, g.SpecRot = l.BandLimit, l.AxisStretch, l.SpecRot
	g.ConeAngle, g.ConeWidth = l.ConeAngle, l.ConeWidth
	g.SpikeCount = 0
	g.Seed += 7001
	if g.phases().blended() {
		g.SeedB += 7001
	}
	g.phaseTo = g.phaseTo.offset(7001)
	return g
}

// standardize shifts and scales a field to zero mean and unit std.
func standardize(f []float64) {
	var mean, sq float64
	for _, v := range f {
		mean += v
	}
	mean /= float64(len(f))
	for _, v := range f {
		sq += (v - mean) * (v - mean)
	}
	sd := math.Sqrt(sq / float64(len(f)))
	if sd < 1e-12 {
		sd = 1
	}
	for i := range f {
		f[i] = (f[i] - mean) / sd
	}
}

// composeLayers combines the main tile a (side n, periodic) with layer B
// per cfg.Layer and returns the composite. a is standardized in place.
func composeLayers(a []float64, n int, cfg Genome) []float64 {
	l := cfg.Layer
	b := synthChannel(n, n, rand.New(rand.NewSource(cfg.Seed+7001)), layerGenome(cfg))
	standardize(a)
	standardize(b)
	mix := clampF(l.Mix, 0, 1)
	out := make([]float64, n*n)
	var mask []float64
	if l.Mode == 1 {
		scale := l.MaskScale
		if scale <= 0 {
			scale = 3
		}
		mask = warpNoise(cfg.Seed+7100, scale, warpSideFor(cfg))
	}
	maskSide := warpSideFor(cfg)
	toM := float64(maskSide) / float64(n)
	parallelRows(n, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			for x := 0; x < n; x++ {
				i := y*n + x
				switch l.Mode {
				case 1:
					m := samplePeriodic(mask, maskSide, float64(x)*toM, float64(y)*toM)
					w := mix / (1 + math.Exp(-l.MaskSharp*(m+l.MaskBias)))
					out[i] = a[i]*(1-w) + b[i]*w
				case 2:
					out[i] = a[i] * (1 + mix*math.Tanh(b[i]))
				case 3:
					out[i] = a[i] + mix*(math.Max(a[i], b[i]+2*l.MaskBias)-a[i])
				default:
					out[i] = a[i]
				}
			}
		}
	})
	return out
}

// samplePeriodic bilinearly samples a periodic n x n field at (x, y).
func samplePeriodic(f []float64, n int, x, y float64) float64 {
	fx, fy := math.Floor(x), math.Floor(y)
	ix, iy := int(fx), int(fy)
	tx, ty := x-fx, y-fy
	mask := n - 1
	x0, y0, x1, y1 := ix&mask, iy&mask, (ix+1)&mask, (iy+1)&mask
	top := f[y0*n+x0]*(1-tx) + f[y0*n+x1]*tx
	bot := f[y1*n+x0]*(1-tx) + f[y1*n+x1]*tx
	return top*(1-ty) + bot*ty
}

// LchPalette is a cosine palette defined in OKLCH, so its genes act on
// perceptual axes and mutate evenly: a step in hue looks like a step in
// hue, a step in lightness like a step in lightness. Over t in [0, 1]:
//
//	L(t) = L0 + LAmp * cos(2*pi*(LFreq*t + LPhase))
//	C(t) = C0 + CAmp * cos(2*pi*(CFreq*t + CPhase))
//	h(t) = H0 + HSpan*t                  (turns: 0.5 = complementary)
//
// Colors outside sRGB are pulled in by chroma at constant L and h.
type LchPalette struct {
	L0     float64 `json:"l0,omitempty"`
	LAmp   float64 `json:"l_amp,omitempty"`
	LFreq  float64 `json:"l_freq,omitempty"`
	LPhase float64 `json:"l_phase,omitempty"`
	C0     float64 `json:"c0,omitempty"`
	CAmp   float64 `json:"c_amp,omitempty"`
	CFreq  float64 `json:"c_freq,omitempty"`
	CPhase float64 `json:"c_phase,omitempty"`
	H0     float64 `json:"h0,omitempty"`
	HSpan  float64 `json:"h_span,omitempty"`
}

func (p LchPalette) at(t float64) (r, g, b float64) {
	L := clampF(p.L0+p.LAmp*math.Cos(2*math.Pi*(p.LFreq*t+p.LPhase)), 0.02, 0.99)
	C := math.Max(0, p.C0+p.CAmp*math.Cos(2*math.Pi*(p.CFreq*t+p.CPhase)))
	return oklchToSRGB(L, C, p.H0+p.HSpan*t)
}

func randomLchPalette(rng *rand.Rand) LchPalette {
	return LchPalette{
		// Lightness stays within ~0.1..0.95 and chroma is vivid enough to
		// read as color; wider ranges produced too many near-black ramps.
		L0: 0.4 + rng.Float64()*0.4, LAmp: 0.1 + rng.Float64()*0.2,
		LFreq: 0.3 + rng.Float64()*1.2, LPhase: rng.Float64(),
		C0: 0.06 + rng.Float64()*0.14, CAmp: rng.Float64() * 0.1,
		CFreq: 0.3 + rng.Float64()*1.7, CPhase: rng.Float64(),
		H0: rng.Float64(), HSpan: rng.Float64()*2 - 1,
	}
}

// randomAnchorColor draws an anchor stop in OKLCH (uniform lightness and
// hue, moderate chroma), which spreads colors far more evenly than uniform
// sRGB cubes, whose samples cluster in muddy mid-tones.
func randomAnchorColor(rng *rand.Rand) [3]float64 {
	r, g, b := oklchToSRGB(0.2+rng.Float64()*0.75, rng.Float64()*0.2, rng.Float64())
	return [3]float64{r, g, b}
}

// MotionSpec is live motion for animations: how a cell image moves on
// its own (holds, single-cell loops) and while it morphs. Every motion
// completes a whole number of cycles per loop, so loops are seamless.
type MotionSpec struct {
	Flow     float64 `json:"flow"`      // 0 = off; ~0.5 gentle, 3 wild: texture churns
	Drift    int     `json:"drift"`     // tile widths traveled per loop, 0 = off
	DriftDir int     `json:"drift_dir"` // 0..7: E, NE, N, NW, W, SW, S, SE
	Color    int     `json:"color"`     // palette cycles per loop, 0 = off
	Morph    float64 `json:"morph"`     // 0 = off; ~0.5 subtle, 3 wild: forms and textures shape-shift
}

func (m MotionSpec) active() bool { return m.Flow > 0 || m.Drift > 0 || m.Color > 0 || m.Morph > 0 }

// churns reports whether m changes g's tile-space structure from frame to
// frame (so a per-cell tile cache cannot be shared across frames).
func (m MotionSpec) churns(g Genome) bool {
	return g.RD.Mode == 0 && (m.Flow > 0 || (m.Morph > 0 && g.LumaRef == ""))
}

// driftDirs are whole-tile lattice steps (screen y points down), so a
// drift of any integer speed returns to the start after one loop.
var driftDirs = [8][2]int{{1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}}

// animateGenome returns g at loop position tau in [0, 1) under motion m.
//
//	Color: the cosine palette's phase (every channel alike) or the OKLCH
//	       hue turns Color times; anchor ramps ping-pong (their end colors
//	       differ, so wrapping would sweep a hard edge through the image).
//	Drift: the texture slides Drift tiles along a lattice direction.
//	Flow:  spectral phases (and Worley points) churn; see flowTurns.
//	Morph: the shape genes (spectral slopes, anisotropy, rotation, band
//	       limit, warp, layer / cellular / LIC genes) each oscillate on
//	       their own cycle while the spectral phases wander (morphPhase),
//	       so forms and textures keep transforming, then return.
//
// Reaction-diffusion genomes skip flow and morph (their chaotic growth
// would turn a changing seed field into flicker), and match-mode genomes
// skip drift and morph (their luminance comes from a canvas-space
// reference, not the tile).
func animateGenome(g Genome, m MotionSpec, tau float64) Genome {
	if m.Color > 0 {
		c := float64(m.Color) * tau
		switch g.PaletteMode {
		case 1:
			g.palShift = 2 * c // ping-pong period is 2
		case 2:
			g.Lch.H0 = frac(g.Lch.H0 + c)
		default:
			for i := range g.PalD {
				g.PalD[i] = frac(g.PalD[i] + c)
			}
		}
	}
	if m.Drift > 0 && g.LumaRef == "" {
		d := driftDirs[((m.DriftDir%8)+8)%8]
		g.driftU = float64(d[0]*m.Drift) * tau
		g.driftV = float64(d[1]*m.Drift) * tau
	}
	if m.Flow > 0 && g.RD.Mode == 0 {
		g.motionFlow, g.motionTau = m.Flow, tau
	}
	if m.Morph > 0 && g.RD.Mode == 0 && g.LumaRef == "" {
		g = morphGenes(g, m.Morph, tau)
		g.motionMorph, g.motionTau = m.Morph, tau
	}
	return g
}

// morphWave is shape gene k's swing at loop position tau, in [-1, 1]: one
// whole cycle per loop at a per-genome phase, minus its value at tau = 0,
// so loops close exactly and a clip starts on the cell itself.
func morphWave(seed int64, k int, tau float64) float64 {
	ph := 2 * math.Pi * cellHash(seed, k, 0, 31)
	return (math.Sin(2*math.Pi*tau+ph) - math.Sin(ph)) / 2
}

// morphGenes swings g's continuous shape genes around their values by
// strength s (amplitudes are per unit of s). Genes that are off stay off,
// and switches that change the render path (cellular-only mix, chroma
// on/off) are never crossed.
func morphGenes(g Genome, s, tau float64) Genome {
	w := func(k int) float64 { return s * morphWave(g.Seed, k, tau) }
	g.Exponent = clampF(g.Exponent+0.25*w(0), 0.5, 10)
	if g.BreakFreq > 0.001 {
		g.ExponentHi = clampF(g.ExponentHi+0.25*w(1), 0.5, 10)
	}
	if g.AxisStretch > 0 {
		g.AxisStretch = clampF(g.AxisStretch*math.Exp(0.15*w(2)), 0.25, 4)
	}
	g.SpecRot += 0.2 * w(3)
	if g.ConeWidth > 0.001 && g.ConeWidth < 0.999 {
		g.ConeAngle += 0.25 * w(4)
	}
	if g.BandLimit > 0 {
		g.BandLimit = clampF(g.BandLimit*math.Exp(0.15*w(5)), 0.01, 1)
	}
	if g.DomainWarp > 0.001 {
		g.DomainWarp = clampF(g.DomainWarp*(1+0.35*w(6)), 0.001, 0.5)
	}
	if g.Gamma > 0 {
		g.Gamma = clampF(g.Gamma*math.Exp(0.06*w(7)), 0.3, 3)
	}
	if g.ReliefStrength > 0 {
		g.ReliefAngle += 0.3 * w(8)
	}
	if g.ChromaStrength > 0.001 {
		g.ChromaStrength = clampF(g.ChromaStrength*(1+0.25*w(9)), 0.002, 0.8)
	}
	if g.Layer.Mode > 0 {
		g.Layer.Exponent = clampF(g.Layer.Exponent+0.25*w(10), 0.5, 10)
		g.Layer.Mix = clampF(g.Layer.Mix+0.12*w(11), 0, 1)
		g.Layer.MaskBias = clampF(g.Layer.MaskBias+0.15*w(12), -1, 1)
	}
	if g.Cell.Mode > 0 {
		g.Cell.Jitter = clampF(g.Cell.Jitter+0.1*w(13), 0, 1)
		if g.Cell.Mix < 0.999 {
			g.Cell.Mix = clampF(g.Cell.Mix+0.1*w(14), 0, 0.99)
		}
	}
	if g.Lic.Mode > 0 {
		g.Lic.Length = clampF(g.Lic.Length*(1+0.2*w(15)), 1, 60)
	}
	return g
}

// morphPhaseAmp is the phase wander (radians per unit strength) a
// shape-shifting bin reaches at most.
const morphPhaseAmp = 0.75

// morphTrig precomputes a shape-shift's per-frame terms for morphPhase.
type morphTrig struct{ c, s float64 }

func newMorphTrig(cfg Genome) morphTrig {
	th := 2 * math.Pi * cfg.motionTau
	a := morphPhaseAmp * cfg.motionMorph / 2
	return morphTrig{a * (math.Cos(th) - 1), a * math.Sin(th)}
}

// morphPhase is bin (x, y)'s phase offset while shape-shifting: each bin
// travels its own ellipse through zero (two random weights on cos - 1 and
// sin), so phases wander without retracing and return after one loop.
// Low frequencies carry most of the energy, so large forms visibly
// transform while fine texture shimmers.
func morphPhase(seed int64, x, y int, t morphTrig) float64 {
	return (cellHash(seed, x, y, 21)*2-1)*t.c + (cellHash(seed, x, y, 22)*2-1)*t.s
}

// flowTurns is how many whole turns spectral bin (x, y) at radial
// frequency f (cycles per tile) makes per loop: more for finer detail, so
// small features churn while large shapes drift slowly, with a random
// sign per bin so the motion boils instead of sliding as a wave. Integer
// turns make the loop seamless.
func flowTurns(x, y int, f, strength float64) float64 {
	k := math.Round(strength * math.Sqrt(f/4))
	if cellHash(1, x, y, 9) < 0.5 {
		k = -k
	}
	return k
}

func frac(x float64) float64 { return x - math.Floor(x) }

// triShift ping-pongs t in [0, 1] by s: a continuous palette walk with
// period 2 (identity at s = 0).
func triShift(t, s float64) float64 {
	if s == 0 {
		return t
	}
	m := t + s
	m -= 2 * math.Floor(m/2)
	if m > 1 {
		m = 2 - m
	}
	return m
}

// phaseSource identifies a phase realization: Seed's random phases,
// optionally rotated toward SeedB's by blend (see Genome.SeedB).
type phaseSource struct {
	seed, seedB int64
	blend       float64
}

func (g Genome) phases() phaseSource {
	return phaseSource{seed: g.Seed, seedB: g.SeedB, blend: g.SeedBlend}
}

// blended reports whether the source actually mixes in a second seed.
func (p phaseSource) blended() bool { return p.seedB != 0 && p.blend > 0 }

// offset shifts both seeds, as the chroma channel does (Seed+9091), so a
// derived channel blends between the matching derived realizations.
func (p phaseSource) offset(off int64) phaseSource {
	if p.seed != 0 {
		p.seed += off
	}
	if p.seedB != 0 {
		p.seedB += off
	}
	return p
}

type SavedGenome struct {
	Version   string `json:"version"`
	CellIndex int    `json:"cell_index"`
	Timestamp string `json:"timestamp"`
	Genome    Genome `json:"genome"`
}

// Hand-picked starting palettes (IQ-style presets) so fresh grids look
// attractive immediately instead of spending generations on the seed lottery.
type Palette struct {
	A, B, C, D [3]float64
}

var palettePresets = []Palette{
	// Rainbow
	{[3]float64{0.5, 0.5, 0.5}, [3]float64{0.5, 0.5, 0.5}, [3]float64{1, 1, 1}, [3]float64{0.0, 0.33, 0.67}},
	// Sunset
	{[3]float64{0.5, 0.5, 0.5}, [3]float64{0.5, 0.5, 0.5}, [3]float64{1, 1, 1}, [3]float64{0.10, 0.25, 0.45}},
	// Dusk (blue-orange)
	{[3]float64{0.5, 0.5, 0.5}, [3]float64{0.5, 0.5, 0.5}, [3]float64{1, 1, 1}, [3]float64{0.0, 0.10, 0.20}},
	// Yellow-pink
	{[3]float64{0.5, 0.5, 0.5}, [3]float64{0.5, 0.5, 0.5}, [3]float64{1, 1, 0.5}, [3]float64{0.8, 0.9, 0.3}},
	// Yellow-green-purple
	{[3]float64{0.5, 0.5, 0.5}, [3]float64{0.5, 0.5, 0.5}, [3]float64{2, 1, 0}, [3]float64{0.5, 0.2, 0.25}},
	// Deep ocean
	{[3]float64{0.66, 0.5, 0.5}, [3]float64{0.5, 0.4, 0.4}, [3]float64{1, 1, 0.8}, [3]float64{0.0, 0.15, 0.3}},
	// Neon
	{[3]float64{0.66, 0.5, 0.5}, [3]float64{0.5, 0.3, 0.4}, [3]float64{1, 1, 1}, [3]float64{0.0, 0.1, 0.2}},
	// Soft pastel
	{[3]float64{0.85, 0.85, 0.85}, [3]float64{0.15, 0.12, 0.1}, [3]float64{1, 1, 1}, [3]float64{0.0, 0.1, 0.2}},
}

func pickPalette(rng *rand.Rand) Palette {
	return palettePresets[rng.Intn(len(palettePresets))]
}

func randomPaletteGenes(rng *rand.Rand) (a, b, c, d [3]float64) {
	for i := 0; i < 3; i++ {
		a[i] = rng.Float64()
		b[i] = 0.3 + rng.Float64()*0.4
		c[i] = rng.Float64()
		d[i] = rng.Float64()
	}
	return
}

func randomGenome(rng *rand.Rand) Genome {
	g := Genome{
		Seed:           rng.Int63(),
		Exponent:       1.5 + rng.Float64()*2.0,
		BandLimit:      0.3 + rng.Float64()*0.6,
		AxisStretch:    0.5 + rng.Float64()*1.5,
		Gamma:          0.5 + rng.Float64()*1.5,
		Colorfulness:   0.4 + rng.Float64()*0.6,
		Transform:      rng.Intn(4),
		TerraceLevels:  3.0 + rng.Float64()*12.0,
		ReliefAngle:    rng.Float64() * 2.0 * math.Pi,
		ReliefStrength: rng.Float64() * rng.Float64() * 1.2, // skewed toward subtle
		ExponentHi:     1.5 + rng.Float64()*2.0,
		BreakFreq:      0.15 + rng.Float64()*0.5,
		SpikeCount:     rng.Intn(5), // 0..4 spikes, 0 = classic
		SpikeAmp:       2.0 + rng.Float64()*14.0,
		ChromaStrength: rng.Float64() * rng.Float64() * 0.5, // skewed toward subtle
		SpecRot:        rng.Float64() * 2.0 * math.Pi,
		ConeAngle:      rng.Float64() * 2.0 * math.Pi,
		ConeWidth:      1.0,                                  // disabled by default; rolled below
		DomainWarp:     rng.Float64() * rng.Float64() * 0.35, // skewed subtle
		WarpScale:      1.5 + rng.Float64()*4.5,
		NormMode:       rng.Intn(3),
		RenderVersion:  renderVersion,
	}
	// 30% of genomes start from a known-good palette
	if rng.Float64() < 0.3 {
		p := pickPalette(rng)
		g.PalA, g.PalB, g.PalC, g.PalD = p.A, p.B, p.C, p.D
	} else {
		g.PalA, g.PalB, g.PalC, g.PalD = randomPaletteGenes(rng)
	}
	// 40% of fresh genomes get a directional cone; the rest stay diffuse.
	if rng.Float64() < 0.4 {
		g.ConeWidth = 0.2 + rng.Float64()*0.6
	}
	if rng.Float64() < 0.3 {
		g.Layer = randomLayer(rng)
	}
	if rng.Float64() < 0.2 {
		g.Cell = randomCell(rng)
	}
	if rng.Float64() < 0.15 {
		g.Lic = randomLic(rng)
	}
	if rng.Float64() < 0.1 {
		g.RD = randomRD(rng)
	}
	g.WarpNest = 1
	if rng.Float64() < 0.4 {
		g.WarpNest = 2
	}
	// Palette family: 30% anchor points, 35% OKLCH cosine, 35% the
	// classic RGB cosine set up above.
	switch r := rng.Float64(); {
	case r < 0.3:
		randomizeAnchors(&g, rng)
	case r < 0.65:
		g.PaletteMode = 2
		g.Lch = randomLchPalette(rng)
	}
	// 25% get k-fold radial symmetry (kaleidoscope / mandala structure).
	if rng.Float64() < 0.25 {
		g.SymmetryFold = 2 + rng.Intn(7)
		g.SymmetryMirror = rng.Float64() < 0.5
	}
	return g
}

// randomizeAnchors switches g to the anchor family with fresh OKLCH stops.
func randomizeAnchors(g *Genome, rng *rand.Rand) {
	g.PaletteMode = 1
	g.AnchorCount = 2 + rng.Intn(4)
	for k := 0; k < g.AnchorCount; k++ {
		g.AnchorColors[k] = randomAnchorColor(rng)
	}
}

func jitter3(v [3]float64, amt float64, rng *rand.Rand) [3]float64 {
	var out [3]float64
	for i := 0; i < 3; i++ {
		out[i] = clampF(v[i]+(rng.Float64()*2-1)*amt, 0.0, 2.0)
	}
	return out
}

// breedGenome creates a child genome that inherits traits from the clicked
// parent (primary, favored) and every locked cell (co-parents). Each
// independent gene and each linked gene group (cone, spikes, palette, ...)
// comes from one genome, favoring the parent (parentShare). The phase
// realization always starts from the parent's and moves by seed blending
// (breedSeed), so children visibly resemble the image you clicked.
//
// strength scales how far the child strays from its parents: every jitter
// amplitude and every discrete-mutation probability is multiplied by it
// (1 = the classic rate). Strong children (>= wildStrength) also always
// take one large gene-group reset.
func breedGenome(parent Genome, donors []Genome, rng *rand.Rand, strength float64) Genome {
	child := Genome{RenderVersion: renderVersion}
	// chance scales a mutation probability by strength, capped at 1.
	chance := func(p float64) bool { return rng.Float64() < math.Min(1, p*strength) }

	// pick chooses which genome one gene (or one linked gene group) comes
	// from: the clicked parent with probability parentShare, otherwise a
	// random locked co-parent.
	pick := func() Genome {
		if len(donors) == 0 || rng.Float64() < parentShare {
			return parent
		}
		return donors[rng.Intn(len(donors))]
	}

	// Independent genes: each one comes from its own pick.
	child.Exponent = pick().Exponent
	child.BandLimit = pick().BandLimit
	child.Gamma = pick().Gamma
	child.Colorfulness = pick().Colorfulness
	child.ChromaStrength = pick().ChromaStrength
	child.NormMode = pick().NormMode

	// Linked gene groups only make sense together (a cone angle without
	// its width, a terrace count without the terrace transform...), so
	// each group comes from ONE genome and is never split across parents.
	g := pick() // anisotropy: rotation only matters with a stretch
	child.AxisStretch, child.SpecRot = g.AxisStretch, g.SpecRot
	g = pick() // spectral breakpoint
	child.ExponentHi, child.BreakFreq = g.ExponentHi, g.BreakFreq
	g = pick() // directional cone
	child.ConeAngle, child.ConeWidth = g.ConeAngle, g.ConeWidth
	g = pick() // spectral spikes
	child.SpikeCount, child.SpikeAmp = g.SpikeCount, g.SpikeAmp
	g = pick() // luminance transform
	child.Transform, child.TerraceLevels = g.Transform, g.TerraceLevels
	g = pick() // relief lighting
	child.ReliefAngle, child.ReliefStrength = g.ReliefAngle, g.ReliefStrength
	g = pick() // domain warp: strength and shape
	child.DomainWarp, child.WarpScale, child.WarpNest = g.DomainWarp, g.WarpScale, g.WarpNest
	child.Layer = pick().Layer // second layer: one linked group
	child.Cell = pick().Cell   // cellular field: one linked group
	child.Lic = pick().Lic     // line integral convolution: one linked group
	child.RD = pick().RD       // reaction-diffusion: one linked group
	g = pick()                 // radial symmetry
	child.SymmetryFold, child.SymmetryMirror = g.SymmetryFold, g.SymmetryMirror
	g = pick() // palette: one coherent set
	child.PalA, child.PalB = g.PalA, g.PalB
	child.PalC, child.PalD = g.PalC, g.PalD
	child.PaletteMode = g.PaletteMode
	child.AnchorCount = g.AnchorCount
	child.AnchorColors = g.AnchorColors
	child.Lch = g.Lch

	// Phase realization: always the clicked parent's, so children visibly
	// resemble the image you clicked.
	child.Seed, child.SeedB, child.SeedBlend = parent.Seed, parent.SeedB, parent.SeedBlend
	breedSeed(&child, donors, rng, strength)

	// --- phase reference inheritance ---
	switch {
	case rng.Float64() < 0.75:
		child.LumaRef, child.LumaW, child.LumaH = parent.LumaRef, parent.LumaW, parent.LumaH
		child.PhaseMix, child.PhaseJitter = parent.PhaseMix, parent.PhaseJitter
		child.Structure, child.Warp = parent.Structure, parent.Warp
		// structural genes: mostly from parent, sometimes recombined
		if rng.Float64() < 0.7 {
			child.Zoom, child.Rot = parent.Zoom, parent.Rot
			child.FlipX, child.FlipY = parent.FlipX, parent.FlipY
			child.CenterX, child.CenterY = parent.CenterX, parent.CenterY
		} else if len(donors) > 0 {
			d := donors[rng.Intn(len(donors))]
			child.Zoom, child.Rot = d.Zoom, d.Rot
			child.FlipX, child.FlipY = d.FlipX, d.FlipY
			child.CenterX, child.CenterY = d.CenterX, d.CenterY
		}
	case len(donors) > 0 && rng.Float64() < 0.5:
		d := donors[rng.Intn(len(donors))]
		child.LumaRef, child.LumaW, child.LumaH = d.LumaRef, d.LumaW, d.LumaH
		child.PhaseMix, child.PhaseJitter = d.PhaseMix, d.PhaseJitter
		// Without these the child got Structure = 0 and the donor's
		// reference dissolved into pure noise in a single generation.
		child.Structure, child.Warp = d.Structure, d.Warp
		child.Zoom, child.Rot = d.Zoom, d.Rot
		child.FlipX, child.FlipY = d.FlipX, d.FlipY
		child.CenterX, child.CenterY = d.CenterX, d.CenterY
	}

	// --- mutation ---
	jitter := func(v, amt float64) float64 {
		return v + (rng.Float64()*2-1)*amt*strength
	}
	// Capped: unbounded, the random walk drifted toward ever-smoother blobs.
	child.Exponent = clampF(jitter(child.Exponent, 0.3), 0.5, 4.5)
	child.BandLimit = clampF(jitter(child.BandLimit, 0.1), 0.01, 1.0)
	child.AxisStretch = clampF(jitter(child.AxisStretch, 0.15), 0.25, 4.0)
	child.Gamma = clampF(jitter(child.Gamma, 0.15), 0.3, 3.0)
	child.Colorfulness = clampF(jitter(child.Colorfulness, 0.1), 0.0, 1.0)
	child.ExponentHi = clampF(jitter(child.ExponentHi, 0.3), 0.5, 10.0)
	child.BreakFreq = clampF(jitter(child.BreakFreq, 0.08), 0.0, 0.9)
	child.TerraceLevels = clampF(jitter(child.TerraceLevels, 1.5), 2.0, 24.0)
	child.ReliefAngle = math.Mod(jitter(child.ReliefAngle, 0.3)+2.0*math.Pi, 2.0*math.Pi)
	child.ReliefStrength = clampF(jitter(child.ReliefStrength, 0.12), 0.0, 2.0)
	if chance(0.12) {
		child.Transform = rng.Intn(4)
	}
	child.SpikeAmp = clampF(jitter(child.SpikeAmp, 2.5), 0.0, 20.0)
	if chance(0.15) {
		child.SpikeCount = rng.Intn(5) // spike count mutates discretely
	}
	child.ChromaStrength = clampF(jitter(child.ChromaStrength, 0.08), 0.0, 0.8)
	child.SpecRot = math.Mod(jitter(child.SpecRot, 0.4)+2*math.Pi, 2*math.Pi)
	child.ConeAngle = math.Mod(jitter(child.ConeAngle, 0.4)+2*math.Pi, 2*math.Pi)
	// Floor at 0.05: below 0.001 the cone switches OFF, so drifting to 0
	// turned the narrowest cone into no cone at all in one step.
	child.ConeWidth = clampF(jitter(child.ConeWidth, 0.1), 0.05, 1.0)
	child.DomainWarp = clampF(jitter(child.DomainWarp, 0.08), 0.0, 0.5)
	child.WarpScale = clampF(jitter(child.WarpScale, 0.4), 1.0, 10.0)
	if child.WarpNest < 1 {
		child.WarpNest = 1
	}
	if chance(0.1) {
		child.WarpNest = 3 - child.WarpNest // 1 <-> 2
	}
	if chance(0.1) {
		child.NormMode = rng.Intn(3)
	}
	if chance(0.1) {
		child.SymmetryFold = rng.Intn(9) // 0..8; <2 disables
	}
	if chance(0.12) {
		child.SymmetryMirror = !child.SymmetryMirror
	}
	if r := &child.RD; r.Mode > 0 {
		// Feed/kill steps are small: the pattern-forming region is narrow.
		r.Feed = clampF(jitter(r.Feed, 0.0015), 0.01, 0.09)
		r.Kill = clampF(jitter(r.Kill, 0.0008), 0.045, 0.07)
		r.Steps = clampi(int(float64(r.Steps)*math.Exp((rng.Float64()*2-1)*0.2*strength)), 100, 6000)
		r.Threshold = clampF(jitter(r.Threshold, 0.1), -1, 2)
		r.Mix = clampF(jitter(r.Mix, 0.08), 0, 1)
		if chance(0.06) { // jump to another pattern family
			p := rdPresets[rng.Intn(len(rdPresets))]
			r.Feed, r.Kill = p[0], p[1]
		}
	}
	if chance(0.04) { // rare: add or drop the reaction-diffusion
		if child.RD.Mode == 0 {
			child.RD = randomRD(rng)
		} else {
			child.RD.Mode = 0
		}
	}
	if l := &child.Lic; l.Mode > 0 {
		l.Length = clampF(l.Length*math.Exp((rng.Float64()*2-1)*0.15*strength), 2, 60)
		l.Mix = clampF(jitter(l.Mix, 0.08), 0, 1)
		l.FlowScale = clampF(jitter(l.FlowScale, 0.4), 1, 10)
		l.Smooth = clampF(jitter(l.Smooth, 1), 1, 40)
		if chance(0.06) {
			l.Mode = 3 - l.Mode // contour <-> swirl
		}
		if chance(0.08) {
			l.Width = clampi(l.Width+1-2*rng.Intn(2), 0, 3) // one step finer or coarser
		}
	}
	if chance(0.05) { // rare: add or drop the streaks
		if child.Lic.Mode == 0 {
			child.Lic = randomLic(rng)
		} else {
			child.Lic.Mode = 0
		}
	}
	if c := &child.Cell; c.Mode > 0 {
		c.Density = clampF(c.Density*math.Exp((rng.Float64()*2-1)*0.12*strength), 2, 48)
		c.Jitter = clampF(jitter(c.Jitter, 0.08), 0, 1)
		c.Mix = clampF(jitter(c.Mix, 0.08), 0, 1)
		if chance(0.08) {
			c.Mode = 1 + (c.Mode+rng.Intn(2))%3 // another cellular mode
		}
		if chance(0.05) {
			c.Metric = (c.Metric + 1 + rng.Intn(2)) % 3
		}
	}
	if chance(0.05) { // rare: add or drop the cellular field
		if child.Cell.Mode == 0 {
			child.Cell = randomCell(rng)
		} else {
			child.Cell.Mode = 0
		}
	}
	if l := &child.Layer; l.Mode > 0 {
		l.Exponent = clampF(jitter(l.Exponent, 0.3), 0.5, 4.5)
		l.ExponentHi = clampF(jitter(l.ExponentHi, 0.3), 0.5, 10.0)
		l.BreakFreq = clampF(jitter(l.BreakFreq, 0.08), 0.0, 0.9)
		l.BandLimit = clampF(jitter(l.BandLimit, 0.1), 0.01, 1.0)
		l.AxisStretch = clampF(jitter(l.AxisStretch, 0.15), 0.25, 4.0)
		l.SpecRot = math.Mod(jitter(l.SpecRot, 0.4)+2*math.Pi, 2*math.Pi)
		l.ConeAngle = math.Mod(jitter(l.ConeAngle, 0.4)+2*math.Pi, 2*math.Pi)
		l.ConeWidth = clampF(jitter(l.ConeWidth, 0.1), 0.05, 1.0)
		l.Mix = clampF(jitter(l.Mix, 0.08), 0.0, 1.0)
		l.MaskScale = clampF(jitter(l.MaskScale, 0.4), 1.0, 8.0)
		l.MaskSharp = clampF(l.MaskSharp*math.Exp((rng.Float64()*2-1)*0.25*strength), 1.0, 40.0)
		l.MaskBias = clampF(jitter(l.MaskBias, 0.1), -1.0, 1.0)
	}
	if chance(0.06) { // rare: add, drop or change the second layer
		switch {
		case child.Layer.Mode == 0:
			child.Layer = randomLayer(rng)
		case rng.Float64() < 0.5:
			child.Layer.Mode = 0
		default:
			child.Layer.Mode = 1 + (child.Layer.Mode+rng.Intn(2))%3 // another mode
		}
	}
	if chance(0.05) { // rare: swap to one of the other two palette families
		switch (child.PaletteMode + 1 + rng.Intn(2)) % 3 {
		case 0:
			child.PaletteMode = 0
		case 1:
			if child.AnchorCount < 2 {
				randomizeAnchors(&child, rng)
			}
			child.PaletteMode = 1
		case 2:
			if child.Lch.L0 == 0 {
				child.Lch = randomLchPalette(rng)
			}
			child.PaletteMode = 2
		}
	}
	switch child.PaletteMode {
	case 1:
		// Anchors mutate on perceptual axes: lightness, chroma and hue
		// steps of the same size look like the same amount of change.
		for k := 0; k < child.AnchorCount && k < 5; k++ {
			c := child.AnchorColors[k]
			L, C, h := srgbToOKLCH(clampF(c[0], 0, 1), clampF(c[1], 0, 1), clampF(c[2], 0, 1))
			r, g, b := oklchToSRGB(clampF(jitter(L, 0.04), 0.02, 0.99), clampF(jitter(C, 0.02), 0, 0.35), jitter(h, 0.03))
			child.AnchorColors[k] = [3]float64{r, g, b}
		}
	case 2:
		p := &child.Lch
		wrap := func(v float64) float64 { return math.Mod(v+2, 1) }
		p.L0 = clampF(jitter(p.L0, 0.04), 0.1, 0.95)
		p.LAmp = clampF(jitter(p.LAmp, 0.03), 0, 0.5)
		p.LFreq = clampF(jitter(p.LFreq, 0.06), 0, 3)
		p.LPhase = wrap(jitter(p.LPhase, 0.04))
		p.C0 = clampF(jitter(p.C0, 0.015), 0, 0.3)
		p.CAmp = clampF(jitter(p.CAmp, 0.01), 0, 0.2)
		p.CFreq = clampF(jitter(p.CFreq, 0.06), 0, 3)
		p.CPhase = wrap(jitter(p.CPhase, 0.04))
		p.H0 = wrap(jitter(p.H0, 0.03))
		p.HSpan = clampF(jitter(p.HSpan, 0.06), -1.5, 1.5)
	}
	if chance(0.08) { // occasionally flip the cone on/off
		if child.ConeWidth > 0.999 {
			child.ConeWidth = 0.2 + rng.Float64()*0.6
		} else {
			child.ConeWidth = 1.0
		}
	}
	child.PalA = jitter3(child.PalA, 0.1*strength, rng)
	child.PalB = jitter3(child.PalB, 0.1*strength, rng)
	child.PalC = jitter3(child.PalC, 0.05*strength, rng)

	if rng.Float64() < 0.7 {
		for i := 0; i < 3; i++ {
			child.PalD[i] = math.Mod(jitter(child.PalD[i], 0.05)+1.0, 1.0)
		}
	} else {
		child.PalD = jitter3(child.PalD, 0.1*strength, rng)
	}

	if child.LumaRef != "" {
		// Structural drift: jitter, occasional flips, gentle reframing.
		child.Structure = clampF(jitter(child.Structure, 0.12)-0.06, 0.0, 1.0)
		// Morph dynamics: siblings spread across the morph timeline — some stay
		// close to the photo, others dive deep into noise. This is what makes
		// the grid read as frames of a morph animation.
		child.Structure = clampF(jitter(child.Structure, 0.30), 0.0, 1.0)
		child.Warp = clampF(jitter(child.Warp, 0.12), 0.0, 0.5)
		child.PhaseJitter = clampF(jitter(child.PhaseJitter, 0.15), 0.0, 1.2)
		child.Zoom = clampF(jitter(child.Zoom, 0.08), 0.9, 2.2)
		child.Rot = math.Mod(jitter(child.Rot, 0.10)+3*math.Pi, 2*math.Pi) - math.Pi
		child.CenterX = clampF(jitter(child.CenterX, 0.04), -0.25, 0.25)
		child.CenterY = clampF(jitter(child.CenterY, 0.04), -0.25, 0.25)
		if chance(0.07) {
			child.FlipX = 1 - child.FlipX
		}
		if chance(0.07) {
			child.FlipY = 1 - child.FlipY
		}
		if chance(0.08) {
			child.PhaseMix = clampF(child.PhaseMix-0.1, 0.3, 1.0)
		}
	}

	// Occasional larger reset of one gene group; wild children always get one.
	if strength >= wildStrength || chance(0.15) {
		switch rng.Intn(8) {
		case 0:
			child.Exponent = 1.5 + rng.Float64()*2.0
		case 1:
			child.BandLimit = 0.2 + rng.Float64()*0.7
		case 2:
			child.AxisStretch = 0.5 + rng.Float64()*2.0
		case 3:
			child.Colorfulness = rng.Float64()
		case 4:
			p := pickPalette(rng)
			child.PalA, child.PalB, child.PalC, child.PalD = p.A, p.B, p.C, p.D
		case 5:
			child.Transform = rng.Intn(4)
			child.TerraceLevels = 3.0 + rng.Float64()*12.0
		case 6: // fresh palette within the current family
			switch child.PaletteMode {
			case 1:
				randomizeAnchors(&child, rng)
			case 2:
				child.Lch = randomLchPalette(rng)
			default:
				child.PalA, child.PalB, child.PalC, child.PalD = randomPaletteGenes(rng)
			}
		case 7:
			if child.LumaRef != "" { // rare: strong step toward randomness
				child.Structure = rng.Float64() * 0.5
			}
		}
	}
	return child
}

// parentShare is the probability that each gene (or linked gene group)
// comes from the clicked parent rather than a locked co-parent. Above 0.5
// so the clicked image stays recognizable however many cells are locked.
const parentShare = 0.7

// breedSeed mutates the child's phase realization (Seed, SeedB, SeedBlend).
// Changing the seed used to be the largest jump in the system: a child with
// a new seed shared nothing with its parent. Now a "seed event" starts a
// blend toward the new seed instead, so the child sits partway between the
// two realizations, and later generations drift along that path.
//
// A genome holds at most one blend partner. A seed event on an already
// blended genome first snaps to whichever seed dominates (moving at most
// half-way), then starts a new blend from there toward the new partner.
func breedSeed(child *Genome, donors []Genome, rng *rand.Rand, strength float64) {
	clearBlend := func() { child.SeedB, child.SeedBlend = 0, 0 }

	// Drift along the current blend path.
	if child.phases().blended() {
		child.SeedBlend = clampF(child.SeedBlend+(rng.Float64()*2-1)*0.08*strength, 0, 1)
	}
	if child.SeedBlend <= 0 {
		clearBlend() // back at Seed exactly
	}

	// A seed event: how often it happens follows strength.
	if rng.Float64() >= math.Min(0.8, 0.45*strength) {
		return
	}
	partner := rng.Int63()
	if len(donors) > 0 && rng.Float64() < 0.5 {
		partner = donors[rng.Intn(len(donors))].Seed
	}
	// Wild children sometimes still take the old hard jump, so the grid
	// can always escape a realization entirely.
	if strength >= wildStrength && rng.Float64() < 0.4 {
		child.Seed = partner
		clearBlend()
		return
	}
	if child.phases().blended() {
		if child.SeedBlend >= 0.5 {
			child.Seed = child.SeedB
		}
		clearBlend()
	}
	if partner == child.Seed || partner == 0 {
		return
	}
	// Initial step toward the partner: about 0.06-0.19 gentle,
	// 0.13-0.38 normal, 0.31-0.94 wild.
	child.SeedB = partner
	child.SeedBlend = clampF(0.25*strength*(0.5+rng.Float64()), 0.02, 0.95)
}

// Mutation strength tiers for the children of one evolve step.
const (
	gentleStrength = 0.5 // refine: small steps, parent seed almost always kept
	normalStrength = 1.0 // the classic breeding rate
	wildStrength   = 2.5 // explore: big steps plus a forced gene-group reset
)

// slotStrengths assigns a mutation strength to each of n child slots, in
// grid order: roughly 3/8 gentle, 2/8 wild, the rest normal (3 / 3 / 2 for
// a full grid of 8 children). Gentle slots come first, so the top of the
// grid refines the clicked image and the bottom explores away from it.
func slotStrengths(n int) []float64 {
	gentle := int(math.Round(float64(n) * 3 / 8))
	wild := int(math.Round(float64(n) * 2 / 8))
	out := make([]float64, n)
	for i := range out {
		switch {
		case i < gentle:
			out[i] = gentleStrength
		case i >= n-wild:
			out[i] = wildStrength
		default:
			out[i] = normalStrength
		}
	}
	return out
}

func clampF(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// ============================================================================
// COLOR SPACES — sRGB / OKLab / OKLCH
// ============================================================================

// srgbToOKLab converts an sRGB color (0..1 per channel) to OKLab.
func srgbToOKLab(r, g, b float64) (L, A, B float64) {
	lin := func(c float64) float64 {
		if c <= 0.04045 {
			return c / 12.92
		}
		return math.Pow((c+0.055)/1.055, 2.4)
	}
	r, g, b = lin(r), lin(g), lin(b)
	l := math.Cbrt(0.4122214708*r + 0.5363325363*g + 0.0514459929*b)
	m := math.Cbrt(0.2119034982*r + 0.6806995451*g + 0.1073969566*b)
	s := math.Cbrt(0.0883024619*r + 0.2817188376*g + 0.6299787005*b)
	return 0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
		1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
		0.0259040371*l + 0.7827717662*m - 0.8086757660*s
}

// oklabToLinear converts OKLab to LINEAR sRGB (may fall outside [0, 1]).
func oklabToLinear(L, A, B float64) (r, g, b float64) {
	l := L + 0.3963377774*A + 0.2158037573*B
	m := L - 0.1055613458*A - 0.0638541728*B
	s := L - 0.0894841775*A - 1.2914855480*B
	l, m, s = l*l*l, m*m*m, s*s*s
	return 4.0767416621*l - 3.3077115913*m + 0.2309699292*s,
		-1.2684380046*l + 2.6097574011*m - 0.3413193965*s,
		-0.0041960863*l - 0.7034186147*m + 1.7076147010*s
}

// linearToSRGB gamma-encodes one clamped linear channel.
func linearToSRGB(c float64) float64 {
	c = clampF(c, 0, 1)
	if c <= 0.0031308 {
		return 12.92 * c
	}
	return 1.055*math.Pow(c, 1/2.4) - 0.055
}

// oklabToSRGB converts OKLab to sRGB in [0, 1], clamping per channel.
func oklabToSRGB(L, A, B float64) (r, g, b float64) {
	lr, lg, lb := oklabToLinear(L, A, B)
	return linearToSRGB(lr), linearToSRGB(lg), linearToSRGB(lb)
}

// oklchToSRGB converts OKLCH (hue in turns) to sRGB, mapping out-of-gamut
// colors into the gamut by reducing chroma at constant lightness and hue,
// so hue never shifts the way per-channel clipping shifts it.
func oklchToSRGB(L, C, h float64) (r, g, b float64) {
	L = clampF(L, 0, 1)
	C = math.Max(C, 0)
	ca, sa := math.Cos(2*math.Pi*h), math.Sin(2*math.Pi*h)
	inGamut := func(c float64) bool {
		lr, lg, lb := oklabToLinear(L, c*ca, c*sa)
		const eps = 1e-6
		return lr >= -eps && lr <= 1+eps && lg >= -eps && lg <= 1+eps && lb >= -eps && lb <= 1+eps
	}
	if !inGamut(C) {
		lo, hi := 0.0, C
		for i := 0; i < 20; i++ {
			mid := (lo + hi) / 2
			if inGamut(mid) {
				lo = mid
			} else {
				hi = mid
			}
		}
		C = lo
	}
	return oklabToSRGB(L, C*ca, C*sa)
}

// srgbToOKLCH converts sRGB to OKLCH with hue in turns [0, 1).
func srgbToOKLCH(r, g, b float64) (L, C, h float64) {
	L, A, B := srgbToOKLab(r, g, b)
	return L, math.Hypot(A, B), math.Mod(math.Atan2(B, A)/(2*math.Pi)+1, 1)
}

// ============================================================================
// COSINE PALETTE
// ============================================================================

// IQ-style cosine palette: color(t) = a + b * cos(2π * (c * t + d))
// t in [0, 1] -> RGB in [0, 255]
func cosinePalette(t float64, cfg Genome) (uint8, uint8, uint8) {
	var out [3]uint8
	for i := 0; i < 3; i++ {
		v := cfg.PalA[i] + cfg.PalB[i]*math.Cos(2*math.Pi*(cfg.PalC[i]*t+cfg.PalD[i]))
		out[i] = uint8(clampF(v, 0.0, 1.0) * 255.0)
	}
	return out[0], out[1], out[2]
}

// ============================================================================
// IMAGE GENERATION — DIRECT SPECTRAL SYNTHESIS
// ============================================================================

// paletteLookup dispatches between the two palette families.
func paletteLookup(t float64, cfg Genome) (uint8, uint8, uint8) {
	if cfg.PaletteMode == 1 {
		return anchorPalette(t, cfg)
	}
	return cosinePalette(t, cfg)
}

// anchorPalette lerps through the genome's anchor colors over t in [0, 1].
// The anchor family reaches subdued multi-stop ramps (film-curve pastels,
// duotones, branded ramps) the 4-vector cosine palette cannot hit.
func anchorPalette(t float64, cfg Genome) (uint8, uint8, uint8) {
	n := cfg.AnchorCount
	if n < 2 {
		n = 2
	}
	if n > 5 {
		n = 5
	}
	pos := triShift(clampF(t, 0, 1), cfg.palShift) * float64(n-1)
	k := int(pos)
	if k >= n-1 {
		k = n - 2
	}
	f := pos - float64(k)
	var out [3]uint8
	for i := 0; i < 3; i++ {
		v := cfg.AnchorColors[k][i]*(1-f) + cfg.AnchorColors[k+1][i]*f
		out[i] = uint8(clampF(v, 0, 1) * 255.0)
	}
	return out[0], out[1], out[2]
}

// smoothstep01 is the classic smooth Hermite step on [0, 1].
func smoothstep01(t float64) float64 {
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}
	return t * t * (3.0 - 2.0*t)
}

// applyTransform reshapes the normalized luminance t in [0, 1] according to
// the genome's Transform gene. These nonlinearities break the Gaussian
// cloudiness of raw spectral noise: turbulence carves sharp dark ridges,
// ridged concentrates them into bright filaments, terraces quantize the
// field into elevation bands.
func applyTransform(t float64, cfg Genome) float64 {
	switch cfg.Transform {
	case 1: // turbulence
		return 1.0 - math.Abs(2.0*t-1.0)
	case 2: // ridged
		v := 1.0 - math.Abs(2.0*t-1.0)
		return v * v
	case 3: // terraces
		levels := cfg.TerraceLevels
		if levels < 2 {
			levels = 2
		}
		scaled := clampF(t, 0, 1) * (levels - 1.0)
		base := math.Floor(scaled)
		frac := scaled - base
		// narrow smooth transition between bands keeps edges soft but defined
		edge := smoothstep01((frac - 0.5) / 0.2)
		return (base + edge) / (levels - 1.0)
	}
	return t
}

// synthChannel generates a single noise field by shaping random complex
// coefficients directly in the frequency domain, then doing ONE inverse FFT.
// (The old path did: white noise -> forward FFT -> shape -> inverse FFT.
// Since white noise already has a flat spectrum, the forward transform was
// wasted work.)
func synthChannel(padW, padH int, rng *rand.Rand, cfg Genome) []float64 {
	data := make([][]complex128, padH)
	for y := range data {
		data[y] = make([]complex128, padW)
	}
	halfW := padW / 2
	stretch := cfg.AxisStretch
	// refHalf / refPadH: the tile that frequency-relative genes (BandLimit,
	// BreakFreq, spike placement) are measured against. Normally this
	// tile; for a scout, the full fieldTile it stands in for.
	refHalf, refPadH := halfW, padH
	if cfg.scoutTile > 0 {
		refHalf, refPadH = fieldTile/2, fieldTile
	}

	// Phase realizations. Each source draws ONE phase per bin, at exactly
	// the same points of the same traversal, so bin (x,y) of every
	// realization pairs with bin (x,y) of the others:
	//   rng / rngSelfB  - this genome's Seed / SeedB (seed blending gene)
	//   rngTo / rngToB  - the animation partner's Seed / SeedB (phaseTo)
	// A source's blended phase is lerped along the shortest arc; the
	// animation then rotates self's phase toward the partner's.
	var rngSelfB, rngTo, rngToB *rand.Rand
	selfBlend := clampF(cfg.SeedBlend, 0.0, 1.0)
	if cfg.phases().blended() {
		rngSelfB = rand.New(rand.NewSource(cfg.SeedB))
	}
	phaseBlend := clampF(cfg.phaseBlend, 0.0, 1.0)
	toBlend := clampF(cfg.phaseTo.blend, 0.0, 1.0)
	if cfg.phaseTo.seed != 0 {
		rngTo = rand.New(rand.NewSource(cfg.phaseTo.seed))
		if cfg.phaseTo.blended() {
			rngToB = rand.New(rand.NewSource(cfg.phaseTo.seedB))
		}
	}
	drawPhase := func(a, b *rand.Rand, blend float64) float64 {
		p := a.Float64() * 2.0 * math.Pi
		if b != nil {
			p = lerpAngle(p, b.Float64()*2.0*math.Pi, blend)
		}
		return p
	}
	mt := newMorphTrig(cfg)

	// Spectral spikes: deterministic from Seed. Both the chosen bin and its
	// Hermitian conjugate are boosted so the real part of the inverse field
	// carries the full periodic energy.
	// Spike bins as a short slice (at most 2*SpikeCount entries, deduped
	// like the map they replace): a linear scan per bin is far cheaper than
	// hashing an [2]int key for every one of ~10^6 bins.
	type spikeBin struct {
		x, y  int
		boost float64
	}
	var spikes []spikeBin
	addSpike := func(x, y int, b float64) {
		for i := range spikes {
			if spikes[i].x == x && spikes[i].y == y {
				spikes[i].boost = b
				return
			}
		}
		spikes = append(spikes, spikeBin{x, y, b})
	}
	if cfg.SpikeCount > 0 && cfg.SpikeAmp > 0.001 && halfW > 1 { // Intn panics on tiny canvases
		sr := rand.New(rand.NewSource(cfg.Seed + 31337))
		for i := 0; i < cfg.SpikeCount; i++ {
			ix := 1 + sr.Intn(refHalf-1)
			iy := sr.Intn(refPadH) - refPadH/2
			if ix >= halfW || iy < -padH/2 || iy >= padH/2 {
				continue // beyond a scout tile's band: invisible there anyway
			}
			wx := ((ix % padW) + padW) % padW
			wy := ((iy % padH) + padH) % padH
			addSpike(wx, wy, cfg.SpikeAmp)
			addSpike((padW-wx)%padW, (padH-wy)%padH, cfg.SpikeAmp)
		}
	}

	// Spectral rotation + directional cone (precomputed factors).
	rot := cfg.SpecRot
	cr, ci := math.Cos(rot), math.Sin(rot)
	coneOn := cfg.ConeWidth > 0.001 && cfg.ConeWidth < 0.999
	halfCone := cfg.ConeWidth * math.Pi

	// Hermitian fill: randomize only the top half of the frequency plane and
	// mirror conjugates into the bottom half. Halves the RNG work and makes
	// the field exactly real by construction (no imaginary part discarded).
	// Cone attenuation is inherited by the mirrored bin, so the cone is
	// effectively two-lobed (theta and theta+pi) — which is what produces
	// the streaky brushed/fabric look rather than a one-sided flow.
	for y := 0; y <= padH/2; y++ {
		ym := (padH - y) % padH
		selfRow := ym == y // rows 0 and padH/2 pair with themselves
		xmax := padW - 1
		if selfRow {
			xmax = halfW
		}
		for x := 0; x <= xmax; x++ {
			xm := (padW - x) % padW
			fu := freqCoord(x, padW)
			fv := freqCoord(y, padH)
			if rot != 0 {
				fu, fv = fu*cr-fv*ci, fu*ci+fv*cr
			}
			fvv := fv * stretch // anisotropy: directional frequency scaling
			f := math.Sqrt(fu*fu + fvv*fvv)
			if f < 0.5 {
				continue // kill DC and the lowest bin
			}
			// Spectral breakpoint: past BreakFreq*maxFreq switch to the
			// high-frequency exponent. This decouples large-scale billows
			// from fine-grain roughness.
			e := cfg.Exponent
			if cfg.BreakFreq > 0.001 {
				corner := float64(refHalf) * cfg.BreakFreq
				if f > corner {
					e = cfg.ExponentHi
				}
			}
			amp := 1.0 / math.Pow(f, e/2.0)
			for i := range spikes {
				if spikes[i].x == x && spikes[i].y == y {
					amp *= spikes[i].boost
					break
				}
			}
			if cfg.BandLimit > 0 {
				maxFreq := float64(refHalf)
				cutoff := maxFreq * cfg.BandLimit
				if f > cutoff {
					falloff := math.Exp(-((f - cutoff) * (f - cutoff)) / (2 * cutoff * cutoff))
					amp *= falloff
				}
			}
			if coneOn {
				ang := math.Atan2(fvv, fu)
				d := math.Abs(math.Mod(ang-cfg.ConeAngle+3*math.Pi, 2*math.Pi) - math.Pi)
				if d > halfCone {
					amp *= math.Exp(-3.0 * (d - halfCone))
				}
			}
			if selfRow && xm == x {
				// Self-conjugate bins (DC/Nyquist corners) must be real.
				data[y][x] = complex(amp, 0)
				continue
			}
			phase := drawPhase(rng, rngSelfB, selfBlend)
			if rngTo != nil {
				// Rotate this bin's phase toward the animation partner's.
				// The conjugate mirror below keeps the field real, so the
				// inverse FFT stays artifact-free at every blend value.
				phase = lerpAngle(phase, drawPhase(rngTo, rngToB, toBlend), phaseBlend)
			}
			if cfg.motionFlow > 0 {
				phase += flowTurns(x, y, f, cfg.motionFlow) * 2 * math.Pi * cfg.motionTau
			}
			if cfg.motionMorph > 0 {
				phase += morphPhase(cfg.Seed, x, y, mt)
			}
			v := complex(amp*math.Cos(phase), amp*math.Sin(phase))
			data[y][x] = v
			data[ym][xm] = complex(real(v), -imag(v))
		}
	}

	fft2d(data, true)
	result := make([]float64, padW*padH)
	for y := 0; y < padH; y++ {
		for x := 0; x < padW; x++ {
			result[y*padW+x] = real(data[y][x])
		}
	}
	return result
}

// synthPair synthesizes two real spectral fields at once (render version
// 6+): A = synthChannel(padW, padH, rngA, cfgA), B = synthChannel(padW,
// padH, rngB, cfgB), for two genomes with the same spectral geometry
// (stretch, rotation, cone, band limit, spikes, scout tile), as the
// luminance genome and its chroma variant have; their exponents (break
// point) and phase streams may differ. Per-bin geometry is computed once;
// each channel's amplitude multiplies the same factors in the same order
// and draws its phases at the same points as synthChannel, so both spectra
// equal the separate ones. Both real fields then come out of ONE complex
// inverse FFT of Z = A + iB (real part a, imaginary part b), half the
// transform work of two FFTs. Only the FFT's rounding differs (~1e-16).
func synthPair(padW, padH int, rngA *rand.Rand, cfgA Genome, rngB *rand.Rand, cfgB Genome) ([]float64, []float64) {
	data := make([][]complex128, padH)
	for y := range data {
		data[y] = make([]complex128, padW)
	}
	halfW := padW / 2
	cfg := cfgA // shared geometry
	stretch := cfg.AxisStretch
	refHalf, refPadH := halfW, padH
	if cfg.scoutTile > 0 {
		refHalf, refPadH = fieldTile/2, fieldTile
	}

	// Per-channel phase sources and exponents (see synthChannel).
	type channel struct {
		cfg                          Genome
		rng, selfB, to, toB          *rand.Rand
		selfBlend, phaseBlend, toBld float64
		mt                           morphTrig
	}
	mk := func(rng *rand.Rand, c Genome) *channel {
		ch := &channel{cfg: c, rng: rng, selfBlend: clampF(c.SeedBlend, 0, 1),
			phaseBlend: clampF(c.phaseBlend, 0, 1), toBld: clampF(c.phaseTo.blend, 0, 1), mt: newMorphTrig(c)}
		if c.phases().blended() {
			ch.selfB = rand.New(rand.NewSource(c.SeedB))
		}
		if c.phaseTo.seed != 0 {
			ch.to = rand.New(rand.NewSource(c.phaseTo.seed))
			if c.phaseTo.blended() {
				ch.toB = rand.New(rand.NewSource(c.phaseTo.seedB))
			}
		}
		return ch
	}
	chans := [2]*channel{mk(rngA, cfgA), mk(rngB, cfgB)}
	drawPhase := func(a, b *rand.Rand, blend float64) float64 {
		p := a.Float64() * 2.0 * math.Pi
		if b != nil {
			p = lerpAngle(p, b.Float64()*2.0*math.Pi, blend)
		}
		return p
	}
	phaseOf := func(ch *channel, x, y int, f float64) float64 {
		phase := drawPhase(ch.rng, ch.selfB, ch.selfBlend)
		if ch.to != nil {
			phase = lerpAngle(phase, drawPhase(ch.to, ch.toB, ch.toBld), ch.phaseBlend)
		}
		if ch.cfg.motionFlow > 0 {
			phase += flowTurns(x, y, f, ch.cfg.motionFlow) * 2 * math.Pi * ch.cfg.motionTau
		}
		if ch.cfg.motionMorph > 0 {
			phase += morphPhase(ch.cfg.Seed, x, y, ch.mt)
		}
		return phase
	}
	exponentOf := func(c Genome, f float64) float64 {
		if c.BreakFreq > 0.001 && f > float64(refHalf)*c.BreakFreq {
			return c.ExponentHi
		}
		return c.Exponent
	}

	// Spikes: same placement as synthChannel (shared by both channels).
	type spikeBin struct {
		x, y  int
		boost float64
	}
	var spikes []spikeBin
	addSpike := func(x, y int, b float64) {
		for i := range spikes {
			if spikes[i].x == x && spikes[i].y == y {
				spikes[i].boost = b
				return
			}
		}
		spikes = append(spikes, spikeBin{x, y, b})
	}
	if cfg.SpikeCount > 0 && cfg.SpikeAmp > 0.001 && halfW > 1 {
		sr := rand.New(rand.NewSource(cfg.Seed + 31337))
		for i := 0; i < cfg.SpikeCount; i++ {
			ix := 1 + sr.Intn(refHalf-1)
			iy := sr.Intn(refPadH) - refPadH/2
			if ix >= halfW || iy < -padH/2 || iy >= padH/2 {
				continue
			}
			wx := ((ix % padW) + padW) % padW
			wy := ((iy % padH) + padH) % padH
			addSpike(wx, wy, cfg.SpikeAmp)
			addSpike((padW-wx)%padW, (padH-wy)%padH, cfg.SpikeAmp)
		}
	}

	rot := cfg.SpecRot
	cr, ci := math.Cos(rot), math.Sin(rot)
	coneOn := cfg.ConeWidth > 0.001 && cfg.ConeWidth < 0.999
	halfCone := cfg.ConeWidth * math.Pi

	for y := 0; y <= padH/2; y++ {
		ym := (padH - y) % padH
		selfRow := ym == y
		xmax := padW - 1
		if selfRow {
			xmax = halfW
		}
		for x := 0; x <= xmax; x++ {
			xm := (padW - x) % padW
			fu := freqCoord(x, padW)
			fv := freqCoord(y, padH)
			if rot != 0 {
				fu, fv = fu*cr-fv*ci, fu*ci+fv*cr
			}
			fvv := fv * stretch
			f := math.Sqrt(fu*fu + fvv*fvv)
			if f < 0.5 {
				continue
			}
			// Shared factors, applied to each channel in synthChannel's order.
			spike, hasSpike := 1.0, false
			for i := range spikes {
				if spikes[i].x == x && spikes[i].y == y {
					spike, hasSpike = spikes[i].boost, true
					break
				}
			}
			falloff, hasFalloff := 1.0, false
			if cfg.BandLimit > 0 {
				cutoff := float64(refHalf) * cfg.BandLimit
				if f > cutoff {
					falloff, hasFalloff = math.Exp(-((f-cutoff)*(f-cutoff))/(2*cutoff*cutoff)), true
				}
			}
			cone, hasCone := 1.0, false
			if coneOn {
				ang := math.Atan2(fvv, fu)
				d := math.Abs(math.Mod(ang-cfg.ConeAngle+3*math.Pi, 2*math.Pi) - math.Pi)
				if d > halfCone {
					cone, hasCone = math.Exp(-3.0*(d-halfCone)), true
				}
			}
			// f^(-e/2) as exp(-e/2 * ln f), one log shared by both channels
			// (cheaper than Pow; differs only in rounding, like the FFT).
			var amps [2]float64
			lf := math.Log(f)
			for c, ch := range chans {
				amp := math.Exp(-exponentOf(ch.cfg, f) / 2.0 * lf)
				if hasSpike {
					amp *= spike
				}
				if hasFalloff {
					amp *= falloff
				}
				if hasCone {
					amp *= cone
				}
				amps[c] = amp
			}
			if selfRow && xm == x {
				data[y][x] = complex(amps[0], amps[1]) // A + iB, both real
				continue
			}
			var va, vb complex128
			for c, ch := range chans {
				sn, cs := math.Sincos(phaseOf(ch, x, y, f))
				v := complex(amps[c]*cs, amps[c]*sn)
				if c == 0 {
					va = v
				} else {
					vb = v
				}
			}
			// Z = A + iB on this bin, conj(A) + i conj(B) on its mirror.
			iB := complex(-imag(vb), real(vb))
			iBc := complex(imag(vb), real(vb)) // i * conj(vb)
			data[y][x] = va + iB
			data[ym][xm] = complex(real(va), -imag(va)) + iBc
		}
	}

	fft2d(data, true)
	a, b := make([]float64, padW*padH), make([]float64, padW*padH)
	parallelRows(padH, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			for x := 0; x < padW; x++ {
				a[y*padW+x], b[y*padW+x] = real(data[y][x]), imag(data[y][x])
			}
		}
	})
	return a, b
}

// applySymmetry folds the luminance field k-fold around the image center by
// polar remapping: the angle is reduced modulo the sector and, with mirror
// enabled, alternate wedges are reflected. Identity when SymmetryFold < 2.
func applySymmetry(lum []float64, width, height, padW int, cfg Genome) []float64 {
	k := cfg.SymmetryFold
	if k < 2 {
		return lum
	}
	out := make([]float64, len(lum))
	cx := float64(width) / 2.0
	cy := float64(height) / 2.0
	sec := 2.0 * math.Pi / float64(k)
	sample := func(x, y float64) float64 {
		var xx, yy float64
		if cfg.RenderVersion >= 5 {
			// Reflect past the edge: continuous texture, no smeared border.
			xx, yy = mirrorCoord(x, width), mirrorCoord(y, height)
		} else {
			xx = clampF(x, 0, float64(width-1))
			yy = clampF(y, 0, float64(height-1))
		}
		x0, y0 := int(xx), int(yy)
		tx, ty := xx-float64(x0), yy-float64(y0)
		x1, y1 := minInt(x0+1, width-1), minInt(y0+1, height-1)
		top := lum[y0*padW+x0]*(1-tx) + lum[y0*padW+x1]*tx
		bot := lum[y1*padW+x0]*(1-tx) + lum[y1*padW+x1]*tx
		return top*(1-ty) + ty*bot
	}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			dx := float64(x) - cx
			dy := float64(y) - cy
			r := math.Hypot(dx, dy)
			a := math.Atan2(dy, dx)
			if a < 0 {
				a += 2 * math.Pi
			}
			w := math.Mod(a, sec)
			if cfg.SymmetryMirror && int(a/sec)%2 == 1 {
				w = sec - w
			}
			sx := cx + math.Cos(w)*r
			sy := cy + math.Sin(w)*r
			out[y*padW+x] = sample(sx, sy)
		}
	}
	return out
}

// tileCacheEntry holds one cell's frame-invariant tile-space results; see
// Genome.tileCache. Filled once (concurrent frame workers share it) and
// read-only afterwards.
type tileCacheEntry struct {
	once      sync.Once
	tw        *tileWarp
	tf        *tileFold
	chroma    []float64
	canonical []float64 // nil for match-mode genomes
}

// specField is the structure stage of a render: the normalized luminance
// (row stride padW, valid region width x height) and the optional chroma
// modulation field (nil when ChromaStrength is off). Rendering is
// colorizeField(spectralField(...)); the animation shape morph warps and
// blends these fields between keyframes before colorizing.
type specField struct {
	width, height, padW int
	lum                 []byte    // normalized field, legacy bytes (render version < 2)
	lumF                []float32 // the same field unquantized, 0..1 (version 2+); may be nil
	chroma              []float64
}

func generateSpectralImage(width, height int, rng *rand.Rand, cfg Genome) *image.RGBA {
	f := spectralField(width, height, rng, cfg)
	var img *image.RGBA
	stage("colorize", func() { img = colorizeField(f, rng, cfg) })
	return img
}

// spectralField synthesizes the luminance structure (spectral noise or
// match-mode reference, domain warp, symmetry, normalization) and the
// chroma field: everything that decides SHAPES, nothing that decides color.
func spectralField(width, height int, rng *rand.Rand, cfg Genome) specField {
	padW := nextPow2(width)
	padH := nextPow2(height)
	tile := fieldTile
	if cfg.scoutTile > 0 {
		tile = cfg.scoutTile
	}
	makeWarp := func() (tw *tileWarp) { // version 3+ domain warp; nil = none
		if cfg.RenderVersion >= 3 {
			stage("warp-fields", func() { tw = newTileWarp(cfg, tile) })
		}
		return tw
	}
	makeFold := func() *tileFold { // version 5+ tile-space symmetry; nil = none / legacy
		if cfg.RenderVersion >= 5 {
			return newTileFold(cfg, tile)
		}
		return nil
	}
	foldedOnTile := false // symmetry already applied while resampling

	// Chroma tile, realized up front so the classic path can resample it
	// together with luminance (resampleTileN shares the fold/warp/footprint
	// work). It has its own rng stream, so the order does not matter.
	chromaCfg := func() Genome {
		chCfg := cfg
		chCfg.Transform = 0
		chCfg.BreakFreq = 0
		chCfg.ReliefStrength = 0
		chCfg.MutationRate = 0
		chCfg.MutationPower = 0
		// The chroma field is realized from Seed+9091, so seed blending
		// and animation phase morphs must target the matching CHROMA
		// realizations, or the chroma field never converges to theirs.
		if chCfg.phases().blended() {
			chCfg.SeedB += 9091
		}
		chCfg.phaseTo = chCfg.phaseTo.offset(9091)
		return chCfg
	}
	padT := nextPow2(tile)
	// Version 6+: when both a spectral structure tile and a chroma tile are
	// needed, synthesize them together (synthPair: one FFT for both).
	usePair := cfg.RenderVersion >= 6 && cfg.ChromaStrength > 0.001 && cfg.LumaRef == "" &&
		!(cfg.Cell.Mode > 0 && cfg.Cell.Mix >= 0.999)
	var pairA, pairB []float64
	getPair := func() {
		if pairA == nil {
			stage("synth", func() {
				pairA, pairB = synthPair(padT, padT, rand.New(rand.NewSource(cfg.Seed)), cfg,
					rand.New(rand.NewSource(cfg.Seed+9091)), chromaCfg())
			})
		}
	}
	makeChroma := func() (chromaRaw []float64) {
		if cfg.ChromaStrength <= 0.001 {
			return nil
		}
		if usePair {
			getPair()
			return pairB
		}
		stage("synth", func() {
			chromaRaw = synthChannel(padT, padT, rand.New(rand.NewSource(cfg.Seed+9091)), chromaCfg())
		})
		return chromaRaw
	}

	// makeCanonical builds the composed structure tile (spectral and/or
	// cellular, layers, reaction-diffusion, LIC) for the classic path.
	makeCanonical := func() []float64 {
		var canonical []float64
		if cfg.Cell.Mode > 0 && cfg.Cell.Mix >= 0.999 {
			stage("cellular", func() { canonical = worleyTile(padT, cfg) }) // cellular only: no spectral FFT
		} else {
			if usePair {
				getPair()
				canonical = pairA
			} else {
				stage("synth", func() { canonical = synthChannel(padT, padT, rand.New(rand.NewSource(cfg.Seed)), cfg) })
			}
			if cfg.Cell.Mode > 0 {
				var cell []float64
				stage("cellular", func() { cell = worleyTile(padT, cfg) })
				standardize(canonical)
				standardize(cell)
				mix := clampF(cfg.Cell.Mix, 0, 1)
				for i := range canonical {
					canonical[i] = canonical[i]*(1-mix) + cell[i]*mix
				}
			}
		}
		if cfg.Layer.Mode > 0 {
			stage("layers", func() { canonical = composeLayers(canonical, padT, cfg) })
		}
		if cfg.RD.Mode > 0 {
			stage("rd", func() { applyRD(canonical, padT, cfg) })
		}
		if cfg.Lic.Mode > 0 {
			stage("lic", func() { applyLic(canonical, padT, cfg) })
		}
		if cfg.motionMorph > 0 {
			// Shape-shifting genes change the field's overall amplitude (a
			// steeper slope is a louder field); normalization is affine-
			// invariant, so standardizing changes nothing but keeps every
			// frame on the range frozen at the cell's first frame.
			standardize(canonical)
		}
		return canonical
	}

	var tw *tileWarp
	var tf *tileFold
	var chromaRaw, chromaResampled, cachedCanonical []float64
	if c := cfg.tileCache; c != nil {
		c.once.Do(func() {
			c.tw, c.tf, c.chroma = makeWarp(), makeFold(), makeChroma()
			if cfg.LumaRef == "" {
				c.canonical = makeCanonical()
			}
		})
		tw, tf, chromaRaw, cachedCanonical = c.tw, c.tf, c.chroma, c.canonical
	} else {
		tw, tf, chromaRaw = makeWarp(), makeFold(), makeChroma()
	}

	var luminance []float64
	if cfg.LumaRef != "" {
		structured := synthMatchChannel(width, height, cfg)
		if structured != nil {
			s := clampF(cfg.Structure, 0, 1)
			if s >= 0.999 {
				luminance = structured
			} else {
				random := synthChannel(padW, padH, rng, cfg)
				// Patchy morph mask: a smooth per-genome noise field shifts the
				// blend weight around its mean, so the dissolve sweeps across
				// the image in patches instead of fading uniformly.
				mr := rand.New(rand.NewSource(cfg.Seed + 4242))
				var mFx, mFy, mPh [3]float64
				for h := 0; h < 3; h++ {
					mFx[h] = 1.0 + mr.Float64()*3.0
					mFy[h] = 1.0 + mr.Float64()*3.0
					mPh[h] = mr.Float64() * 2 * math.Pi
				}
				for y := 0; y < height; y++ {
					for x := 0; x < width; x++ {
						ux, uy, _, _ := warpFrame(x, y, width, height, cfg)
						d := 0.0
						for h := 0; h < 3; h++ {
							d += math.Sin(2*math.Pi*(mFx[h]*ux+mFy[h]*uy) + mPh[h])
						}
						w := clampF(s+0.30*(d/3.0), 0.0, 1.0)
						i := y*padW + x
						structured[i] = structured[i]*w + random[i]*(1.0-w)
					}
				}
				luminance = structured
			}
		}
	}
	if luminance == nil { // classic mode or broken ref
		// Universal-tile synthesis: the field is realized ONCE on a fixed
		// periodic grid defined purely by the genome, then resampled to the
		// requested canvas. Every resolution samples the same realization,
		// so exports are exactly scaled versions of the grid previews (and
		// symmetric/kaleidoscope structures keep their seams aligned).
		canonical := cachedCanonical
		if canonical == nil {
			canonical = makeCanonical()
		}
		if chromaRaw != nil {
			stage("resample", func() {
				both := resampleTileN([][]float64{canonical, chromaRaw}, tile, padT, width, height, padW, height, tw, tf,
					cfg.driftU*float64(tile), cfg.driftV*float64(tile))
				luminance, chromaResampled = both[0], both[1]
			})
		} else {
			stage("resample", func() {
				luminance = resampleTileN([][]float64{canonical}, tile, padT, width, height, padW, height, tw, tf,
					cfg.driftU*float64(tile), cfg.driftV*float64(tile))[0]
			})
		}
		foldedOnTile = tf != nil
	} else if tw != nil {
		// Match mode: the same tile-space flow, mapped onto the canvas.
		luminance = warpCanvas(luminance, width, height, padW, tile, tw)
	}

	// Legacy domain warp (render version < 3): displace the canvas field
	// through three seed-driven sines before normalization.
	if cfg.RenderVersion < 3 && cfg.DomainWarp > 0.001 {
		luminance = warpDomain(luminance, width, height, padW, cfg)
	}

	// Radial symmetry on the canvas: all modes before version 5, and match
	// mode (which has no tile) from version 5 on.
	if cfg.SymmetryFold >= 2 && !foldedOnTile {
		luminance = applySymmetry(luminance, width, height, padW, cfg)
	}

	var normLum []byte
	var normLumF []float32
	nc := normCtl{fixed: cfg.normFix, lo: cfg.normLo, hi: cfg.normHi, capture: cfg.normCapture,
		quant: cfg.normQuant, captureQuant: cfg.normCaptureQuant}
	stage("normalize", func() { normLum, normLumF = normalize255(luminance, cfg.NormMode, width, height, padW, nc) })

	// Chroma modulation field: an independent spectral realization mapped to
	// [-1, 1]. Added to the palette coordinate t per pixel, it produces
	// spatially structured color shifts (iridescence) the static cosine
	// palette cannot express.
	var chromaField []float64
	if chromaRaw != nil {
		cmn, cmx := minMax(chromaRaw)
		crange := cmx - cmn
		if crange < 1e-10 {
			crange = 1
		}
		chromaField = chromaResampled
		if chromaField == nil { // match mode: no shared tile pass
			chromaField = resampleTile(chromaRaw, tile, nextPow2(tile), width, height, padW, padH, tw, tf)
		}
		for i := range chromaField {
			chromaField[i] = (chromaField[i]-cmn)/crange*2.0 - 1.0
		}
	}

	return specField{width: width, height: height, padW: padW, lum: normLum, lumF: normLumF, chroma: chromaField}
}

// colorizeField turns a structure field into the final image: transform,
// relief lighting, gamma, chroma shift, palette, colorfulness, mutation.
func colorizeField(f specField, rng *rand.Rand, cfg Genome) *image.RGBA {
	if cfg.RenderVersion >= 2 {
		return colorizeFloat(f, rng, cfg)
	}
	width, height, padW := f.width, f.height, f.padW
	normLum, chromaField := f.lum, f.chroma

	// Relief shading resolution normalization: adjacent pixels sample the
	// field at intervals proportional to wpp, so the per-pixel gradient
	// weakens on large canvases. Scale it back to the grid preview's world
	// slope (long side 256 reference) so relief looks identical everywhere.
	maxSide := width
	if height > maxSide {
		maxSide = height
	}
	reliefGain := float64(maxSide) / 256.0
	reliefSpan := int(reliefGain + 0.5)
	if reliefSpan < 1 {
		reliefSpan = 1
	}

	rgba := image.NewRGBA(image.Rect(0, 0, width, height))
	shade := func(x, y int) (r, g, b uint8) {
		idx := y*padW + x

		lum01 := applyTransform(float64(normLum[idx])/255.0, cfg)
		if cfg.ReliefStrength > 0.001 && x > reliefSpan && x < width-1-reliefSpan && y > reliefSpan && y < height-reliefSpan {
			// Gradient measured over reliefSpan pixels keeps the world-space
			// sampling distance constant across resolutions (matches the
			// 256px preview's footprint) and suppresses byte-quantization
			// noise that reliefGain would otherwise amplify into grain.
			dxl := (float64(normLum[idx+reliefSpan]) - float64(normLum[idx-reliefSpan])) / (510.0 * float64(reliefSpan))
			dyl := (float64(normLum[idx+reliefSpan*padW]) - float64(normLum[idx-reliefSpan*padW])) / (510.0 * float64(reliefSpan))
			lx := math.Cos(cfg.ReliefAngle)
			ly := math.Sin(cfg.ReliefAngle)
			lum01 = clampF(lum01+cfg.ReliefStrength*reliefGain*(dxl*lx+dyl*ly), 0, 1)
		}
		t := math.Pow(lum01, cfg.Gamma)
		if chromaField != nil {
			t = clampF(t+cfg.ChromaStrength*chromaField[idx], 0, 1)
		}
		r, g, b = paletteLookup(t, cfg)

		// v1+: desaturate toward the palette color's own luma. Legacy
		// genomes mixed in the raw field instead, which with a transform
		// or relief overlaid a second, untransformed structure.
		gray := normLum[idx]
		if cfg.RenderVersion >= 1 {
			gray = clampByte(int(0.2126*float64(r) + 0.7152*float64(g) + 0.0722*float64(b) + 0.5))
		}
		cf := cfg.Colorfulness
		r = clampByte(int(float64(gray)*(1.0-cf) + float64(r)*cf))
		g = clampByte(int(float64(gray)*(1.0-cf) + float64(g)*cf))
		b = clampByte(int(float64(gray)*(1.0-cf) + float64(b)*cf))
		return r, g, b
	}
	put := func(x, y int, r, g, b uint8) {
		o := rgba.PixOffset(x, y)
		rgba.Pix[o], rgba.Pix[o+1], rgba.Pix[o+2], rgba.Pix[o+3] = r, g, b, 255
	}

	if cfg.MutationRate > 0 {
		// Mutation noise draws from rng per pixel: must stay sequential.
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				r, g, b := shade(x, y)
				if rng.Float64() < cfg.MutationRate {
					mutation := int(randNorm(0, cfg.MutationPower, rng))
					r = clampByte(int(r) + mutation)
					g = clampByte(int(g) + mutation)
					b = clampByte(int(b) + mutation)
				}
				put(x, y, r, g, b)
			}
		}
		return rgba
	}

	// No mutation: pixels are independent, shade them across all cores.
	parallelRows(height, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			for x := 0; x < width; x++ {
				r, g, b := shade(x, y)
				put(x, y, r, g, b)
			}
		}
	})
	// Leave the caller's rng exactly where the sequential loop would (one
	// draw per pixel): reverse-engineering searches reuse one rng across
	// many renders, so skipping the draws would change their results.
	if rng != nil {
		for i := 0; i < width*height; i++ {
			rng.Float64()
		}
	}
	return rgba
}

// paletteLUTSize is the resolution of the float palette table. Palettes
// are smooth in t (at most a few cosine cycles), so linear interpolation
// between 1024 samples is far below one 8-bit step of error.
const paletteLUTSize = 1024

// paletteLUT samples the genome's palette over t in [0, 1] as float sRGB.
// Building it once per render keeps per-pixel cost flat even for the
// OKLCH family's gamut mapping.
type paletteLUT [paletteLUTSize + 1][3]float64

func buildPaletteLUT(cfg Genome) *paletteLUT {
	var lut paletteLUT
	// Anchor stops, converted once (render version 2 lerps in OKLab).
	n := clampi(cfg.AnchorCount, 2, 5)
	var stops [5][3]float64
	for k := 0; k < n; k++ {
		c := cfg.AnchorColors[k]
		stops[k][0], stops[k][1], stops[k][2] = srgbToOKLab(clampF(c[0], 0, 1), clampF(c[1], 0, 1), clampF(c[2], 0, 1))
	}
	for i := range lut {
		t := float64(i) / paletteLUTSize
		var r, g, b float64
		switch cfg.PaletteMode {
		case 1:
			pos := triShift(t, cfg.palShift) * float64(n-1)
			k := minInt(int(pos), n-2)
			f := pos - float64(k)
			var lab [3]float64
			for c := 0; c < 3; c++ {
				lab[c] = stops[k][c]*(1-f) + stops[k+1][c]*f
			}
			r, g, b = oklabToSRGB(lab[0], lab[1], lab[2])
		case 2:
			r, g, b = cfg.Lch.at(t)
		default:
			var v [3]float64
			for c := 0; c < 3; c++ {
				v[c] = clampF(cfg.PalA[c]+cfg.PalB[c]*math.Cos(2*math.Pi*(cfg.PalC[c]*t+cfg.PalD[c])), 0, 1)
			}
			r, g, b = v[0], v[1], v[2]
		}
		lut[i] = [3]float64{r, g, b}
	}
	return &lut
}

func (lut *paletteLUT) at(t float64) (r, g, b float64) {
	pos := clampF(t, 0, 1) * paletteLUTSize
	i := minInt(int(pos), paletteLUTSize-1)
	f := pos - float64(i)
	a, c := lut[i], lut[i+1]
	return a[0] + (c[0]-a[0])*f, a[1] + (c[1]-a[1])*f, a[2] + (c[2]-a[2])*f
}

// bayer8 is the 8x8 ordered-dither threshold matrix (values 0..63).
var bayer8 = [8][8]float64{
	{0, 32, 8, 40, 2, 34, 10, 42},
	{48, 16, 56, 24, 50, 18, 58, 26},
	{12, 44, 4, 36, 14, 46, 6, 38},
	{60, 28, 52, 20, 62, 30, 54, 22},
	{3, 35, 11, 43, 1, 33, 9, 41},
	{51, 19, 59, 27, 49, 17, 57, 25},
	{15, 47, 7, 39, 13, 45, 5, 37},
	{63, 31, 55, 23, 61, 29, 53, 21},
}

// quantizeDither converts a float channel in [0, 1] to a byte with an
// ordered dither: sub-LSB precision becomes a fine, deterministic pattern
// instead of visible 8-bit bands in smooth gradients.
func quantizeDither(v float64, x, y int) uint8 {
	return clampByte(int(math.Floor(v*255 + (bayer8[y&7][x&7]+0.5)/64)))
}

// colorizeFloat is colorizeField for render version 2+: the whole chain
// (transform, relief, gamma, chroma shift, palette, colorfulness) runs in
// float, and the result is quantized once, dithered. Per-pixel mutation
// noise is not applied (no version 2 genome carries it).
func colorizeFloat(f specField, rng *rand.Rand, cfg Genome) *image.RGBA {
	width, height, padW := f.width, f.height, f.padW
	lum := f.lumF
	if lum == nil { // fields built outside spectralField: derive from bytes
		lum = make([]float32, len(f.lum))
		for i, v := range f.lum {
			lum[i] = float32(v) / 255
		}
	}
	lut := buildPaletteLUT(cfg)

	// Relief: same world-scale normalization as the legacy path, but the
	// gradient comes from the float field, so there is no quantization
	// noise for reliefGain to amplify.
	reliefGain := float64(maxInt(width, height)) / 256.0
	reliefSpan := maxInt(int(reliefGain+0.5), 1)
	lx, ly := math.Cos(cfg.ReliefAngle), math.Sin(cfg.ReliefAngle)
	relief := cfg.ReliefStrength > 0.001

	rgba := image.NewRGBA(image.Rect(0, 0, width, height))
	parallelRows(height, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			for x := 0; x < width; x++ {
				idx := y*padW + x
				lum01 := applyTransform(float64(lum[idx]), cfg)
				if relief && x > reliefSpan && x < width-1-reliefSpan && y > reliefSpan && y < height-reliefSpan {
					dxl := float64(lum[idx+reliefSpan]-lum[idx-reliefSpan]) / (2 * float64(reliefSpan))
					dyl := float64(lum[idx+reliefSpan*padW]-lum[idx-reliefSpan*padW]) / (2 * float64(reliefSpan))
					lum01 = clampF(lum01+cfg.ReliefStrength*reliefGain*(dxl*lx+dyl*ly), 0, 1)
				}
				t := math.Pow(lum01, cfg.Gamma)
				if f.chroma != nil {
					t = clampF(t+cfg.ChromaStrength*f.chroma[idx], 0, 1)
				}
				r, g, b := lut.at(t)
				cf := cfg.Colorfulness
				gray := 0.2126*r + 0.7152*g + 0.0722*b
				r, g, b = gray+(r-gray)*cf, gray+(g-gray)*cf, gray+(b-gray)*cf
				o := rgba.PixOffset(x, y)
				rgba.Pix[o] = quantizeDither(r, x, y)
				rgba.Pix[o+1] = quantizeDither(g, x, y)
				rgba.Pix[o+2] = quantizeDither(b, x, y)
				rgba.Pix[o+3] = 255
			}
		}
	})
	// Leave the caller's rng where the legacy path would (one draw per
	// pixel), so reverse-engineering searches sharing an rng stay stable.
	if rng != nil {
		for i := 0; i < width*height; i++ {
			rng.Float64()
		}
	}
	return rgba
}

// fieldTile is the side of the universal synthesis tile. The classic field
// is ALWAYS realized on this fixed periodic grid — for previews, exports,
// and animation alike — then resampled to the canvas. This is what makes
// the saved image an exact scaled version of the grid cell. Raise it for
// crisper large exports; lower it if evolution renders feel slow.
const fieldTile = 1024

// resampleTile maps the fixed periodic tile onto a canvas ISOTROPICALLY:
// one uniform world-per-pixel scale anchored to the canvas long side, so
// canvases of different aspect ratios sample the same geometry with the
// same stretch (circles stay circles; symmetry seams stay put). The tile
// wraps periodically beyond its bounds, matching the FFT field. When the
// canvas samples the tile coarsely (downscaling), the footprint is
// supersampled into a low-pass average to prevent aliasing; when sampling
// finely (upscaling), a single bilinear tap gives smooth interpolation.
//
// warp (render version 3+, may be nil) displaces every tile-space sample
// coordinate before lookup, so the domain warp is computed in tile space:
// identical at every resolution and seamless across the periodic tile.
//
// fold (render version 5+, may be nil) folds each sample coordinate k-fold
// around the tile center before the warp: the tile-space form of radial
// symmetry, which never samples outside real texture.
func resampleTile(src []float64, tile, srcStride, dw, dh, dstStride, dstRows int, warp *tileWarp, fold *tileFold) []float64 {
	return resampleTileN([][]float64{src}, tile, srcStride, dw, dh, dstStride, dstRows, warp, fold, 0, 0)[0]
}

// multiSampler bilinearly samples every field at tile coordinate (u, v)
// into out[k]: the index and weights are computed once and shared, and
// each field's value is the exact expression the single-field sampler
// used, so results are bit-identical to sampling the fields separately.
type multiSampler func(u, v float64, out []float64)

// Only the valid dw x dh region is computed; the power-of-two padding
// (up to 2.4x the pixels for a 768x576 preview canvas) is left zero,
// since every consumer (normalize255, colorize, symmetry, legacy warp)
// reads the valid region alone.
//
// resampleTileN is resampleTile for several fields of the same tile size
// (luminance and chroma) in ONE pass: the coordinate transform (fold,
// warp, footprint) is computed once per pixel and shared by every field.
// Per field, taps are summed in the same order as before, so each output
// is bit-identical to a separate resampleTile call.
//
// offU, offV (tile units) translate every final sample coordinate, after
// fold and warp: the live-motion drift, a rigid slide of the periodic
// texture under the frame. Zero for still renders.
func resampleTileN(srcs [][]float64, tile, srcStride, dw, dh, dstStride, dstRows int, warp *tileWarp, fold *tileFold, offU, offV float64) [][]float64 {
	nf := len(srcs)
	outs := make([][]float64, nf)
	for k := range outs {
		outs[k] = make([]float64, dstStride*dstRows)
	}
	long := dw
	if dh > long {
		long = dh
	}
	wpp := float64(tile) / float64(long) // tile units per canvas pixel (isotropic)
	ft := float64(tile)

	var samp multiSampler = func(u, v float64, out []float64) {
		// math.Mod returns its argument unchanged for 0 <= u < ft (its
		// reduction loop never runs), and canvas coordinates almost always
		// lie inside the tile, so skipping it there is exact and avoids
		// the call that dominated unwarped resampling.
		if u < 0 || u >= ft {
			u = math.Mod(u, ft)
			if u < 0 {
				u += ft
			}
		}
		if v < 0 || v >= ft {
			v = math.Mod(v, ft)
			if v < 0 {
				v += ft
			}
		}
		iu, iv := int(u), int(v)
		fu, fv := u-float64(iu), v-float64(iv)
		ju, jv := (iu+1)%tile, (iv+1)%tile
		for k, src := range srcs {
			a := src[iv*srcStride+iu]
			b := src[iv*srcStride+ju]
			c := src[jv*srcStride+iu]
			d := src[jv*srcStride+ju]
			out[k] = (a*(1-fu)+b*fu)*(1-fv) + (c*(1-fu)+d*fu)*fv
		}
	}

	// Version 4+ fast sampler: wrap with floor and a power-of-two mask
	// instead of math.Mod (the hot spot once the warp is interpolated).
	// The tile is always a power of two (fieldTile, scoutTileSide).
	fast := samp
	if (fold != nil || (warp != nil && warp.interp)) && tile&(tile-1) == 0 {
		mask := tile - 1
		fast = func(u, v float64, out []float64) {
			fu, fv := math.Floor(u), math.Floor(v)
			iu, iv := int(fu), int(fv)
			tu, tv := u-fu, v-fv
			x0, y0 := iu&mask, iv&mask
			x1, y1 := (iu+1)&mask, (iv+1)&mask
			for k, src := range srcs {
				a := src[y0*srcStride+x0]
				b := src[y0*srcStride+x1]
				c := src[y1*srcStride+x0]
				d := src[y1*srcStride+x1]
				out[k] = (a*(1-tu)+b*tu)*(1-tv) + (c*(1-tu)+d*tu)*tv
			}
		}
	}
	if offU != 0 || offV != 0 {
		slow, quick := samp, fast
		samp = func(u, v float64, out []float64) { slow(u+offU, v+offV, out) }
		fast = func(u, v float64, out []float64) { quick(u+offU, v+offV, out) }
	}
	if fold != nil {
		// Folded (version 5+): every coordinate goes through fold then
		// warp. Pixel centers are transformed once and taps interpolated,
		// except on fold seams, where the exact footprint's Jacobian probe
		// spans two sectors and raises the tap count (anti-aliasing the
		// seam) instead of blending coordinates of different sectors.
		xf := fold.apply
		if warp != nil {
			xf = func(u, v float64) (float64, float64) { return warp.apply(fold.apply(u, v)) }
		}
		resampleWarped(outs, fast, xf, ft, wpp, dw, dh, dstStride, dstRows, true)
		return outs
	}
	if warp != nil && warp.interp {
		resampleWarped(outs, fast, warp.apply, ft, wpp, dw, dh, dstStride, dstRows, false)
		return outs
	}
	ns := int(wpp) + 1                  // supersampling taps per axis (>=1)
	parallelRows(dh, func(ya, yb int) { // valid rows only: padding is never read
		acc, tmp := make([]float64, nf), make([]float64, nf)
		for y := ya; y < yb; y++ {
			ly := y % dh
			oy := ft/2 + (float64(ly)+0.5-float64(dh)/2)*wpp
			for x := 0; x < dw; x++ {
				lx := x % dw
				ox := ft/2 + (float64(lx)+0.5-float64(dw)/2)*wpp
				if warp != nil {
					footprintN(warp.apply, samp, ox, oy, wpp, acc, tmp)
				} else if ns == 1 {
					samp(ox, oy, acc)
				} else {
					step := wpp / float64(ns)
					for k := range acc {
						acc[k] = 0
					}
					for ky := 0; ky < ns; ky++ {
						vv := oy - wpp/2 + (float64(ky)+0.5)*step
						for kx := 0; kx < ns; kx++ {
							uu := ox - wpp/2 + (float64(kx)+0.5)*step
							samp(uu, vv, tmp)
							for k := range acc {
								acc[k] += tmp[k]
							}
						}
					}
					for k := range acc {
						acc[k] /= float64(ns * ns)
					}
				}
				for k := range outs {
					outs[k][y*dstStride+x] = acc[k]
				}
			}
		}
	})
	return outs
}

// warpFieldSide is the resolution of the periodic warp noise fields. The
// warp is smooth (a handful of cycles per tile), so 256 samples bilinearly
// interpolated are exact to well below a pixel at any canvas size.
const warpFieldSide = 256

// scoutWarpSide is the warp/flow/mask noise resolution for scouts: the
// warp spectrum is negligible (e^-18) well below 64^2's 32-cycle Nyquist,
// and scouts are statistical stand-ins anyway (they already draw other
// phases than the preview). Building 256^2 fields was 10% of Generate All.
const scoutWarpSide = 64

// warpSideFor is the smooth-noise field side a render uses.
func warpSideFor(cfg Genome) int {
	if cfg.scoutTile > 0 {
		return scoutWarpSide
	}
	return warpFieldSide
}

// tileWarp is the render version 3 domain warp: four smooth, periodic
// spectral noise fields over the synthesis tile (seeds Seed+5150..5153),
// each normalized to unit standard deviation, displacing tile-space
// sample coordinates. Being periodic on the tile, the flow wraps without
// seams; being defined in tile units, it is the same at every resolution
// and aspect ratio, and the field never smears at the canvas border.
type tileWarp struct {
	fields [4][]float64
	pairs  [2][][2]float64 // fields 2k and 2k+1 interleaved: one lookup per displacement
	amp    float64         // displacement std in tile units
	// interp (render version 4+): resampleTile warps each canvas pixel
	// center once and interpolates sub-pixel taps and the footprint
	// Jacobian from neighboring centers (resampleWarped), instead of
	// warping every tap plus four Jacobian probes (warpedFootprint).
	interp bool
	toF    float64 // tile units -> field samples
	nest   int
	n      int     // field side (power of two)
	nf     float64 // float64(n)
	invN   float64 // 1/n, exact for a power of two (x*invN == x/n)
}

func newTileWarp(cfg Genome, tile int) *tileWarp {
	if cfg.DomainWarp <= 0.001 {
		return nil
	}
	scale := cfg.WarpScale
	if scale <= 0 {
		scale = 3
	}
	w := &tileWarp{
		// 0.4 matches the old sine warp's displacement std per unit of
		// DomainWarp, so the gene keeps its strength meaning.
		amp:    0.4 * clampF(cfg.DomainWarp, 0, 0.5) * float64(tile),
		toF:    float64(warpSideFor(cfg)) / float64(tile),
		n:      warpSideFor(cfg),
		nf:     float64(warpSideFor(cfg)),
		invN:   1 / float64(warpSideFor(cfg)),
		nest:   clampi(cfg.WarpNest, 1, 2),
		interp: cfg.RenderVersion >= 4,
	}
	n := 2 * w.nest
	parallelMap(n, func(i int) { w.fields[i] = warpNoise(cfg.Seed+5150+int64(i), scale, w.n) })
	for k := 0; k < w.nest; k++ {
		w.pairs[k] = make([][2]float64, w.n*w.n)
		for i := range w.pairs[k] {
			w.pairs[k][i] = [2]float64{w.fields[2*k][i], w.fields[2*k+1][i]}
		}
	}
	return w
}

// warpNoise synthesizes one periodic warp field: random phases under a
// 1/f amplitude spectrum with a Gaussian cutoff at scale cycles per tile,
// normalized to zero mean and unit standard deviation.
func warpNoise(seed int64, scale float64, n int) []float64 {
	rng := rand.New(rand.NewSource(seed))
	data := make([][]complex128, n)
	for y := range data {
		data[y] = make([]complex128, n)
		for x := range data[y] {
			fu, fv := freqCoord(x, n), freqCoord(y, n)
			f := math.Hypot(fu, fv)
			ph := rng.Float64() * 2 * math.Pi
			if f < 0.5 {
				continue
			}
			amp := math.Exp(-0.5*(f/scale)*(f/scale)) / f
			data[y][x] = complex(amp*math.Cos(ph), amp*math.Sin(ph))
		}
	}
	fft2d(data, true)
	out := make([]float64, n*n)
	var mean, sq float64
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			v := real(data[y][x])
			out[y*n+x] = v
			mean += v
		}
	}
	mean /= float64(n * n)
	for i := range out {
		out[i] -= mean
		sq += out[i] * out[i]
	}
	if sd := math.Sqrt(sq / float64(n*n)); sd > 1e-12 {
		for i := range out {
			out[i] /= sd
		}
	}
	return out
}

// resampleWarped is the render version 4 warped resample. The warp is
// smooth at pixel scale (25+ canvas pixels per cycle even on previews), so
// warped coordinates are computed once per pixel center (plus a one-pixel
// border) and everything else is interpolated from them: the footprint's
// source span comes from the neighbors' central differences, and each
// sub-pixel tap bilinearly interpolates the warped centers around it. Tap
// counts follow the same rule as warpedFootprint. Each worker keeps a
// rolling three-row window of warped centers, so memory stays flat at any
// canvas size.
//
// xf is the coordinate transform (the warp; fold then warp from version 5).
// With seams set, the transform may jump (non-mirrored fold seams): a
// pixel whose neighboring centers are farther than seamSpan pixel spans
// away on either side is rendered with the exact per-tap footprint, which
// is always correct, so the threshold trades only speed.
func resampleWarped(outs [][]float64, samp multiSampler, xf func(u, v float64) (float64, float64),
	ft, wpp float64, dw, dh, dstStride, dstRows int, seams bool) {
	nf := len(outs)
	type pt struct{ u, v float64 }
	center := func(l, n int) float64 { return ft/2 + (float64(l)+0.5-float64(n)/2)*wpp }
	mapRow := func(ly int, row []pt) {
		for i := range row { // row[i] is canvas column lx = i-1
			row[i].u, row[i].v = xf(center(i-1, dw), center(ly, dh))
		}
	}
	parallelRows(dh, func(ya, yb int) { // valid rows only: padding is never read
		rows := [3][]pt{make([]pt, dw+2), make([]pt, dw+2), make([]pt, dw+2)}
		key := [3]int{-2, -2, -2} // canvas row held in each buffer
		acc, tmp := make([]float64, nf), make([]float64, nf)
		put := func(i int) {
			for k := range outs {
				outs[k][i] = acc[k]
			}
		}
		want := 0 // canvas row currently being produced
		get := func(ly int) []pt {
			for i := range key {
				if key[i] == ly {
					return rows[i]
				}
			}
			// Evict a buffer not needed by the current output row; callers
			// request rows cur-1, cur, cur+1 with cur = the row being made,
			// so pass it in via want.
			slot := -1
			for i := range key {
				if key[i] < want-1 || key[i] > want+1 {
					slot = i
					break
				}
			}
			mapRow(ly, rows[slot])
			key[slot] = ly
			return rows[slot]
		}
		for y := ya; y < yb; y++ {
			ly := y % dh
			want = ly
			up, mid, dn := get(ly-1), get(ly), get(ly+1)
			for x := 0; x < dw; x++ {
				i := x%dw + 1
				c := mid[i]
				if seams {
					lim := seamSpan * wpp
					d := func(p, q pt) float64 { return math.Hypot(p.u-q.u, p.v-q.v) }
					if d(mid[i+1], c) > lim || d(c, mid[i-1]) > lim || d(dn[i], c) > lim || d(c, up[i]) > lim {
						footprintN(xf, samp, center(x%dw, dw), center(ly, dh), wpp, acc, tmp)
						put(y*dstStride + x)
						continue
					}
				}
				lenX := math.Hypot(mid[i+1].u-mid[i-1].u, mid[i+1].v-mid[i-1].v) / 2
				lenY := math.Hypot(dn[i].u-up[i].u, dn[i].v-up[i].v) / 2
				nx := clampi(int(lenX)+1, 1, maxWarpTaps)
				ny := clampi(int(lenY)+1, 1, maxWarpTaps)
				if nx == 1 && ny == 1 {
					samp(c.u, c.v, acc)
					put(y*dstStride + x)
					continue
				}
				for k := range acc {
					acc[k] = 0
				}
				for ky := 0; ky < ny; ky++ {
					sy := (float64(ky)+0.5)/float64(ny) - 0.5 // pixel units, -0.5..0.5
					r0, r1, fy := up, mid, sy+1
					if sy >= 0 {
						r0, r1, fy = mid, dn, sy
					}
					for kx := 0; kx < nx; kx++ {
						sx := (float64(kx)+0.5)/float64(nx) - 0.5
						j0, fx := i-1, sx+1
						if sx >= 0 {
							j0, fx = i, sx
						}
						a, b, cc, d := r0[j0], r0[j0+1], r1[j0], r1[j0+1]
						u := (a.u*(1-fx)+b.u*fx)*(1-fy) + (cc.u*(1-fx)+d.u*fx)*fy
						v := (a.v*(1-fx)+b.v*fx)*(1-fy) + (cc.v*(1-fx)+d.v*fx)*fy
						samp(u, v, tmp)
						for k := range acc {
							acc[k] += tmp[k]
						}
					}
				}
				for k := range acc {
					acc[k] /= float64(nx * ny)
				}
				put(y*dstStride + x)
			}
		}
	})
}

// tileFold is radial symmetry in tile space (render version 5+): a sample
// coordinate is folded k-fold around the tile center (the canvas center
// maps there), mirroring alternate sectors when mirror is set. Same fold
// as applySymmetry, but on the periodic tile, so the folded point always
// lands on real texture instead of a clamped canvas border.
type tileFold struct {
	sec    float64 // sector angle, 2*pi/k
	c      float64 // tile center
	mirror bool
}

func newTileFold(cfg Genome, tile int) *tileFold {
	if cfg.SymmetryFold < 2 {
		return nil
	}
	return &tileFold{sec: 2 * math.Pi / float64(cfg.SymmetryFold), c: float64(tile) / 2, mirror: cfg.SymmetryMirror}
}

func (f *tileFold) apply(u, v float64) (float64, float64) {
	dx, dy := u-f.c, v-f.c
	r := math.Hypot(dx, dy)
	a := math.Atan2(dy, dx)
	if a < 0 {
		a += 2 * math.Pi
	}
	w := math.Mod(a, f.sec)
	if f.mirror && int(a/f.sec)%2 == 1 {
		w = f.sec - w
	}
	sw, cw := math.Sincos(w)
	return f.c + cw*r, f.c + sw*r
}

// seamSpan: a folded pixel whose neighbor centers map more than this many
// pixel spans away is treated as straddling a seam (exact footprint).
// Smooth warps rarely stretch this far; when they do, the exact path is
// still correct, just slower.
const seamSpan = 4.0

// maxWarpTaps caps the per-axis supersampling of one warped pixel; fold
// lines of a strong nested warp compress space without bound.
const maxWarpTaps = 12

// warpedFootprint box-filters one canvas pixel (center (ox, oy) in tile
// units, wpp tile units wide) through the warp. Where the warp compresses
// space, the source area under the pixel grows and fixed supersampling
// aliases into moiré, so the tap count per axis follows the local warp
// Jacobian: enough taps to cover the warped footprint at about one tap
// per tile unit (exactly the unwarped count when the warp is identity).
func warpedFootprint(warp *tileWarp, samp func(u, v float64) float64, ox, oy, wpp float64) float64 {
	return footprint(warp.apply, samp, ox, oy, wpp)
}

// footprint is warpedFootprint for any coordinate transform xf (warp,
// fold, or fold then warp), single field.
func footprint(xf func(u, v float64) (float64, float64), samp func(u, v float64) float64, ox, oy, wpp float64) float64 {
	var acc, tmp [1]float64
	footprintN(xf, func(u, v float64, out []float64) { out[0] = samp(u, v) }, ox, oy, wpp, acc[:], tmp[:])
	return acc[0]
}

// footprintN box-filters one canvas pixel through xf for every field the
// sampler serves, writing the per-field averages into acc (tmp is
// scratch of the same length). Tap counts follow the local Jacobian; see
// warpedFootprint.
func footprintN(xf func(u, v float64) (float64, float64), samp multiSampler, ox, oy, wpp float64, acc, tmp []float64) {
	h := wpp / 2
	ax, ay := xf(ox+h, oy)
	bx, by := xf(ox-h, oy)
	cx, cy := xf(ox, oy+h)
	dx, dy := xf(ox, oy-h)
	lenX := math.Hypot(ax-bx, ay-by) // source span of the pixel's x extent
	lenY := math.Hypot(cx-dx, cy-dy)
	nx := clampi(int(lenX)+1, 1, maxWarpTaps)
	ny := clampi(int(lenY)+1, 1, maxWarpTaps)
	for k := range acc {
		acc[k] = 0
	}
	for ky := 0; ky < ny; ky++ {
		vv := oy - h + (float64(ky)+0.5)*wpp/float64(ny)
		for kx := 0; kx < nx; kx++ {
			uu := ox - h + (float64(kx)+0.5)*wpp/float64(nx)
			tu, tv := xf(uu, vv)
			samp(tu, tv, tmp)
			for k := range acc {
				acc[k] += tmp[k]
			}
		}
	}
	for k := range acc {
		acc[k] /= float64(nx * ny)
	}
}

// at samples field i at tile coordinate (u, v), periodically.
func (w *tileWarp) at(i int, u, v float64) float64 {
	// Exact for power-of-two n: x*invN equals x/n bit for bit, and on the
	// non-negative reduced coordinates &(n-1) equals %n.
	n, mask := w.n, w.n-1
	x, y := u*w.toF, v*w.toF
	x -= math.Floor(x*w.invN) * w.nf
	y -= math.Floor(y*w.invN) * w.nf
	x0, y0 := int(x)&mask, int(y)&mask
	fx, fy := x-math.Floor(x), y-math.Floor(y)
	x1, y1 := (x0+1)&mask, (y0+1)&mask
	f := w.fields[i]
	top := f[y0*n+x0]*(1-fx) + f[y0*n+x1]*fx
	bot := f[y1*n+x0]*(1-fx) + f[y1*n+x1]*fx
	return top*(1-fy) + bot*fy
}

// atPair samples fields 2k and 2k+1 together at (u, v). It is at() for
// both fields with the index and weights computed once and the two values
// fetched from one interleaved slot; per field the arithmetic is the same
// expression as at(), so results are bit-identical.
func (w *tileWarp) atPair(k int, u, v float64) (float64, float64) {
	// Exact for power-of-two n: x*invN equals x/n bit for bit, and on the
	// non-negative reduced coordinates &(n-1) equals %n.
	n, mask := w.n, w.n-1
	x, y := u*w.toF, v*w.toF
	x -= math.Floor(x*w.invN) * w.nf
	y -= math.Floor(y*w.invN) * w.nf
	x0, y0 := int(x)&mask, int(y)&mask
	fx, fy := x-math.Floor(x), y-math.Floor(y)
	x1, y1 := (x0+1)&mask, (y0+1)&mask
	p := w.pairs[k]
	a, b, c, d := p[y0*n+x0], p[y0*n+x1], p[y1*n+x0], p[y1*n+x1]
	top0 := a[0]*(1-fx) + b[0]*fx
	bot0 := c[0]*(1-fx) + d[0]*fx
	top1 := a[1]*(1-fx) + b[1]*fx
	bot1 := c[1]*(1-fx) + d[1]*fx
	return top0*(1-fy) + bot0*fy, top1*(1-fy) + bot1*fy
}

// apply displaces tile coordinate p: p + k*q(p), or with nesting
// p + k*r(p + k*q(p)).
func (w *tileWarp) apply(u, v float64) (float64, float64) {
	qu, qv := w.atPair(0, u, v)
	du, dv := w.amp*qu, w.amp*qv
	if w.nest >= 2 {
		ru, rv := w.atPair(1, u+du, v+dv)
		du, dv = w.amp*ru, w.amp*rv
	}
	return u + du, v + dv
}

// warpCanvas applies a tileWarp to a canvas field (match mode, which has
// no tile): each pixel maps to tile space exactly as resampleTile frames
// it, is displaced there, and maps back. Samples beyond the canvas clamp.
func warpCanvas(lum []float64, width, height, padW, tile int, w *tileWarp) []float64 {
	long := float64(maxInt(width, height))
	wpp := float64(tile) / long
	ft := float64(tile)
	out := make([]float64, len(lum))
	parallelRows(height, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			for x := 0; x < width; x++ {
				u := ft/2 + (float64(x)+0.5-float64(width)/2)*wpp
				v := ft/2 + (float64(y)+0.5-float64(height)/2)*wpp
				wu, wv := w.apply(u, v)
				sx := clampF((wu-ft/2)/wpp+float64(width)/2-0.5, 0, float64(width-1))
				sy := clampF((wv-ft/2)/wpp+float64(height)/2-0.5, 0, float64(height-1))
				x0, y0 := int(sx), int(sy)
				x1, y1 := minInt(x0+1, width-1), minInt(y0+1, height-1)
				tx, ty := sx-float64(x0), sy-float64(y0)
				top := lum[y0*padW+x0]*(1-tx) + lum[y0*padW+x1]*tx
				bot := lum[y1*padW+x0]*(1-tx) + lum[y1*padW+x1]*tx
				out[y*padW+x] = top*(1-ty) + bot*ty
			}
		}
	})
	return out
}

// normalize255 maps the valid width×height sub-region (row stride padW)
// of the field to [0, 255]. Only valid pixels contribute to the statistics,
// so results are identical at every canvas size regardless of how much
// power-of-two padding the buffer carries. It returns both the legacy
// byte field (truncated exactly as it always was: render versions < 2
// read it) and the same values unquantized as float32 in [0, 1] (render
// version 2+). Padding stays zero in both; the render loops never read it.
// normCtl optionally fixes the min-max / percentile range (fixed) and/or
// reports the range used (capture). Zero value = classic behavior.
type normCtl struct {
	fixed   bool
	lo, hi  float64
	capture *[2]float64
	// Rank mode: with fixed and a quant table, values map through the
	// frozen table instead of being re-ranked (no 8M-value sort per
	// animation frame, no rank reshuffling); captureQuant receives the
	// table of a regular rank pass.
	quant        []float64
	captureQuant *[]float64
}

// rankQuantiles is the frozen rank table size: 4097 quantiles map ranks to
// well under one 8-bit level.
const rankQuantiles = 4096

func normalize255(data []float64, mode int, width, height, padW int, nc normCtl) ([]byte, []float32) {
	result := make([]byte, len(data))
	resultF := make([]float32, len(data))
	if width <= 0 || height <= 0 {
		return result, resultF
	}
	// Direct path for min-max (and any frozen min-max / percentile range):
	// map straight from the strided field into the outputs, skipping the
	// compact copy and the intermediate array (~130 MB per FullHD frame).
	// Same min/max over the same pixels and the same per-pixel expression
	// as normalizeRegion, so the results are identical.
	if mode == 0 || (nc.fixed && mode == 1) {
		var lo, hi float64
		if nc.fixed {
			lo, hi = nc.lo, nc.hi
		} else {
			lo, hi = minMaxStrided(data, width, height, padW)
		}
		if nc.capture != nil {
			nc.capture[0], nc.capture[1] = lo, hi
		}
		rangeVal := hi - lo
		if rangeVal < 1e-10 {
			rangeVal = 1
		}
		parallelRows(height, func(ya, yb int) {
			for y := ya; y < yb; y++ {
				for i := y * padW; i < y*padW+width; i++ {
					v := (data[i] - lo) / rangeVal * 255.0
					if v < 0 {
						v = 0
					}
					if v > 255 {
						v = 255
					}
					result[i], resultF[i] = byte(v), float32(v/255.0)
				}
			}
		})
		return result, resultF
	}
	// Fast path: no padding at all — normalize the buffer directly.
	if width == padW && height*padW == len(data) {
		norm := normalizeRegion(data, mode, nc)
		parallelRows(len(norm), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				v := norm[i]
				result[i], resultF[i] = byte(v), float32(v/255.0)
			}
		})
		return result, resultF
	}

	// Extract the valid region as a compact contiguous copy so the
	// statistics (min/max, percentiles, ranks) see only real pixels.
	// Copies and per-pixel mapping run in parallel: element-wise, so the
	// results are identical to the serial loops.
	valid := make([]float64, width*height)
	parallelRows(height, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			copy(valid[y*width:(y+1)*width], data[y*padW:y*padW+width])
		}
	})
	norm := normalizeRegion(valid, mode, nc)
	parallelRows(height, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			for x := 0; x < width; x++ {
				v := norm[y*width+x]
				result[y*padW+x], resultF[y*padW+x] = byte(v), float32(v/255.0)
			}
		}
	})
	return result, resultF
}

// normalizeRegion maps a compact (unpadded) float field to [0, 255],
// unquantized. mode selects the mapping:
// 0 = min-max stretch (classic), 1 = 1st/99th percentile clip,
// 2 = rank equalization (full histogram flattening).
// Truncating its output with byte() reproduces the original byte
// normalization exactly (same expressions, same order).
func normalizeRegion(data []float64, mode int, nc normCtl) []float64 {
	result := make([]float64, len(data))
	if len(data) == 0 {
		return result
	}
	if mode == 2 {
		if nc.fixed && len(nc.quant) > 1 {
			q := nc.quant
			nq := float64(len(q) - 1)
			parallelRows(len(data), func(a, b int) {
				for i := a; i < b; i++ {
					v := data[i]
					k := sort.SearchFloat64s(q, v) // first q[k] >= v
					var pos float64
					switch {
					case k == 0:
						pos = 0
					case k >= len(q):
						pos = nq
					default:
						if d := q[k] - q[k-1]; d > 0 {
							pos = float64(k-1) + (v-q[k-1])/d
						} else {
							pos = float64(k)
						}
					}
					result[i] = pos / nq * 255.0
				}
			})
			return result
		}
		idx := make([]int, len(data))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return data[idx[a]] < data[idx[b]] })
		den := float64(len(data) - 1)
		if den < 1 {
			den = 1
		}
		for rank, i := range idx {
			result[i] = float64(rank) / den * 255.0
		}
		if nc.captureQuant != nil {
			q := make([]float64, rankQuantiles+1)
			for k := range q {
				q[k] = data[idx[int(math.Round(float64(k)*den/rankQuantiles))]]
			}
			*nc.captureQuant = q
		}
		return result
	}
	var lo, hi float64
	if nc.fixed {
		lo, hi = nc.lo, nc.hi
	} else {
		lo, hi = minMaxParallel(data)
		if mode == 1 {
			sorted := append([]float64(nil), data...)
			sort.Float64s(sorted)
			lo = sorted[int(float64(len(sorted)-1)*0.01)]
			hi = sorted[int(float64(len(sorted)-1)*0.99)]
		}
	}
	if nc.capture != nil {
		nc.capture[0], nc.capture[1] = lo, hi
	}
	rangeVal := hi - lo
	if rangeVal < 1e-10 {
		rangeVal = 1
	}
	parallelRows(len(data), func(a, b int) {
		for i := a; i < b; i++ {
			normalized := (data[i] - lo) / rangeVal * 255.0
			if normalized < 0 {
				normalized = 0
			}
			if normalized > 255 {
				normalized = 255
			}
			result[i] = normalized
		}
	})
	return result
}

// minMaxStrided is minMax over the valid width x height region of a
// strided field, across all cores (exact in any order).
func minMaxStrided(data []float64, width, height, padW int) (float64, float64) {
	var mu sync.Mutex
	mn, mx := data[0], data[0]
	parallelRows(height, func(ya, yb int) {
		lo, hi := data[ya*padW], data[ya*padW]
		for y := ya; y < yb; y++ {
			l, h := minMax(data[y*padW : y*padW+width])
			if l < lo {
				lo = l
			}
			if h > hi {
				hi = h
			}
		}
		mu.Lock()
		if lo < mn {
			mn = lo
		}
		if hi > mx {
			mx = hi
		}
		mu.Unlock()
	})
	return mn, mx
}

// minMaxParallel is minMax across all cores (min and max are exact, so the
// result is identical in any order).
func minMaxParallel(data []float64) (float64, float64) {
	if len(data) < 1<<16 {
		return minMax(data)
	}
	var mu sync.Mutex
	mn, mx := data[0], data[0]
	parallelRows(len(data), func(a, b int) {
		lo, hi := minMax(data[a:b])
		mu.Lock()
		if lo < mn {
			mn = lo
		}
		if hi > mx {
			mx = hi
		}
		mu.Unlock()
	})
	return mn, mx
}

// warpDomain displaces the classic-mode luminance field through a smooth,
// seed-driven displacement field (identity at DomainWarp 0), producing
// marbled / flowing structures that raw spectral noise cannot reach.
// Displacement is a fraction of the frame, sampled bilinearly with clamped
// edges.
// warpFrame returns the normalized coordinates of pixel (x, y) and the
// pixel scale of a unit offset in each axis. Legacy genomes normalize each
// axis by its own length (anisotropic, anchored at the top-left corner),
// so the pattern stretched with the canvas aspect ratio. v1+ normalizes
// both axes by the long side around the center, matching resampleTile's
// isotropic centered framing: every aspect sees the same pattern.
func warpFrame(x, y, width, height int, cfg Genome) (ux, uy, sx, sy float64) {
	if cfg.RenderVersion < 1 {
		return float64(x) / float64(width), float64(y) / float64(height), float64(width), float64(height)
	}
	long := float64(maxInt(width, height))
	return (float64(x) - float64(width)/2) / long, (float64(y) - float64(height)/2) / long, long, long
}

func warpDomain(lum []float64, width, height, padW int, cfg Genome) []float64 {
	warp := clampF(cfg.DomainWarp, 0, 0.5)
	wr := rand.New(rand.NewSource(cfg.Seed + 5150))
	var fx, fy, ph [3]float64
	for h := 0; h < 3; h++ {
		fx[h] = 1.0 + wr.Float64()*3.0
		fy[h] = 1.0 + wr.Float64()*3.0
		ph[h] = wr.Float64() * 2 * math.Pi
	}
	sample := func(x, y float64) float64 {
		xx := clampF(x, 0, float64(width-1))
		yy := clampF(y, 0, float64(height-1))
		x0, y0 := int(xx), int(yy)
		tx, ty := xx-float64(x0), yy-float64(y0)
		x1, y1 := minInt(x0+1, width-1), minInt(y0+1, height-1)
		top := lum[y0*padW+x0]*(1-tx) + lum[y0*padW+x1]*tx
		bot := lum[y1*padW+x0]*(1-tx) + lum[y1*padW+x1]*tx
		return top*(1-ty) + ty*bot
	}
	out := make([]float64, len(lum))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			ux, uy, sx, sy := warpFrame(x, y, width, height, cfg)
			wd, wq := 0.0, 0.0
			for h := 0; h < 3; h++ {
				wd += math.Sin(2*math.Pi*(fx[h]*ux+fy[h]*uy) + ph[h])
				wq += math.Sin(2*math.Pi*(fy[h]*ux+fx[h]*uy) + ph[(h+1)%3])
			}
			ox := warp * (wd / 3.0) * sx
			oy := warp * (wq / 3.0) * sy
			out[y*padW+x] = sample(float64(x)+ox, float64(y)+oy)
		}
	}
	return out
}

func clampByte(v int) byte {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(v)
}

func randNorm(mean, stddev float64, rng *rand.Rand) float64 {
	u1 := rng.Float64()
	u2 := rng.Float64()
	z := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	return mean + stddev*z
}

// renderClean renders a genome deterministically with the stochastic
// per-pixel mutation noise disabled. Grid previews and exports both go
// through this, so any size renders the same pixels; only interpolation
// detail differs between resolutions. The genome is copied, never modified.
func renderClean(g Genome, width, height int) *image.RGBA {
	rc := g
	rc.MutationRate = 0
	rc.MutationPower = 0
	return generateSpectralImage(width, height, rand.New(rand.NewSource(rc.Seed)), rc)
}

// renderPreviewFramed renders a genome at the grid preview's field of
// view, supersampled 2x for a smooth, anti-aliased export. The tile is
// sampled isotropically anchored to the canvas long side, so any
// canvas narrower than 4:3 is first narrowed (keeping height) to match
// the 256x192 preview's world window. Rendering that canvas at 2x
// covers the IDENTICAL world window (wpp halves as the canvas doubles),
// so the result can be Lanczos-downsampled back to the requested size —
// suppressing the fine grain a near-1:1 display of the 1024-tile
// otherwise shows. Supersampling is skipped when the doubled canvas
// would exceed the 8192-px render budget (memory), falling back to the
// plain render: at those sizes the field buffer alone is ~0.5 GB.
func renderPreviewFramed(g Genome, width, height int) *image.RGBA {
	if maxW := height * 4 / 3; width > maxW {
		width = maxW
	}
	ss := supersampleFactor(g, fieldTile, width, height)
	if ss == 1 {
		return renderClean(g, width, height)
	}
	big := renderClean(g, width*ss, height*ss)
	return downsampleImage(big, width, height)
}

// renderExactFramed renders a genome at EXACTLY the requested canvas size,
// any aspect ratio, supersampled 2x when the memory budget allows. Unlike
// renderPreviewFramed it does NOT narrow the canvas to the grid preview's
// 4:3 field of view: wide canvases simply capture a wider window of the
// periodic world field (same center), tall canvases a taller one. No
// cropping, no letterboxing — animation frames come out pixel-exact at the
// requested dimensions.
func renderExactFramed(g Genome, width, height int) *image.RGBA {
	ss := supersampleFactor(g, fieldTile, width, height)
	if ss == 1 {
		return renderClean(g, width, height)
	}
	big := renderClean(g, width*ss, height*ss)
	var img *image.RGBA
	stage("downsample", func() { img = downsampleImage(big, width, height) })
	return img
}

// supersampleFactor picks the render scale for a width x height canvas
// sampling a tile of the given side. Render version 4+ scales until each
// sub-pixel spans at most ~4/3 tile units, so colorization (whose palette
// can be steep) sees the field at near its own resolution instead of a
// pre-averaged footprint: 3x for 256px grid previews, 2x for scouts and
// for anything at least ~768px wide. 3x is where the grain is gone by
// eye; 4x and 8x look the same. Older versions keep the fixed 2x.
// Either way the scaled canvas must fit the 8192-px render budget.
func supersampleFactor(g Genome, tile, width, height int) int {
	ss := 2
	if g.RenderVersion >= 4 {
		wpp := float64(tile) / float64(maxInt(width, height))
		ss = clampi(int(math.Ceil(wpp*3/4)), 2, 4)
	}
	for ss > 1 && (width*ss > 8192 || height*ss > 8192) {
		ss--
		if g.RenderVersion < 4 {
			ss = 1 // legacy: 2x or nothing
		}
	}
	return ss
}

// renderGridPreview renders a genome's grid cell. Render version 4+
// previews are supersampled like exports (see supersampleFactor); older
// genomes keep the plain render they always showed.
func renderGridPreview(g Genome) *image.RGBA {
	if g.RenderVersion >= 4 {
		return renderExactFramed(g, imgW, imgH)
	}
	return renderClean(g, imgW, imgH)
}

// cropToPreviewFOV centrally trims a render to the 4:3 field of view the
// grid preview shows (the synthesis tile is sampled isotropically, so
// wider canvases would otherwise capture a wider slice of the field).
// Keeps full height. 1920x1080 -> 1440x1080, 7680x4320 -> 5760x4320.
func cropToPreviewFOV(img *image.RGBA) *image.RGBA {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	target := w
	if max43 := h * 4 / 3; w > max43 {
		target = max43
	}
	if target >= w {
		return img
	}
	x0 := (w - target) / 2
	out := image.NewRGBA(image.Rect(0, 0, target, h))
	draw.Draw(out, out.Bounds(), img, image.Point{X: x0, Y: 0}, draw.Src)
	return out
}

func imgToBase64(img *image.RGBA) string {
	var buf bytes.Buffer
	pngFast.Encode(&buf, img)
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

// ============================================================================
// IMAGE UTILITIES (for reverse engineering)
// ============================================================================

// lanczos2Kernel is the 2-lobe Lanczos window: sharp without the ringing
// of 3-lobe, far better than box averaging for spectrum analysis input.
func lanczos2Kernel(x float64) float64 {
	x = math.Abs(x)
	if x < 1e-8 {
		return 1
	}
	if x >= 2 {
		return 0
	}
	px := math.Pi * x
	return 2 * math.Sin(px) * math.Sin(px/2) / (px * px)
}

func clampi(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// downsampleImage uses Lanczos-2 for strong shrinks (sharper comparison
// targets improve exponent fits and match references) and box averaging
// otherwise.
func downsampleImage(src image.Image, targetW, targetH int) *image.RGBA {
	if src.Bounds().Dx() >= 2*targetW && src.Bounds().Dy() >= 2*targetH {
		return downsampleLanczos2(src, targetW, targetH)
	}
	return downsampleBox(src, targetW, targetH)
}

// downsampleLanczos2 is a separable Lanczos-2 resize.
func downsampleLanczos2(src image.Image, targetW, targetH int) *image.RGBA {
	srcB := src.Bounds()
	srcW := srcB.Dx()
	srcH := srcB.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	if srcW < 1 || srcH < 1 || targetW < 1 || targetH < 1 {
		return dst
	}
	if srcW == targetW && srcH == targetH {
		for y := 0; y < targetH; y++ {
			for x := 0; x < targetW; x++ {
				r, g, b, _ := src.At(srcB.Min.X+x, srcB.Min.Y+y).RGBA()
				dst.SetRGBA(x, y, color.RGBA{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), 255})
			}
		}
		return dst
	}

	xscale := float64(srcW) / float64(targetW)
	yscale := float64(srcH) / float64(targetH)
	xf := math.Max(1.0, xscale)
	yf := math.Max(1.0, yscale)

	// Filter taps depend only on the output column (horizontal pass) or
	// row (vertical pass), so they are computed once per column/row
	// instead of once per pixel; the sin-based kernel dominated the cost.
	// Taps and summation order are unchanged, so output is bit-identical.
	type taps struct {
		idx []int
		w   []float64
	}
	makeTaps := func(n, srcN int, scale, f float64) []taps {
		out := make([]taps, n)
		for x := 0; x < n; x++ {
			center := (float64(x)+0.5)*scale - 0.5
			lo := int(math.Ceil(center - 2.0*f - 0.5))
			hi := int(math.Floor(center + 2.0*f - 0.5))
			for i := lo; i <= hi; i++ {
				d := (float64(i) + 0.5 - center - 0.5) / f
				out[x].idx = append(out[x].idx, clampi(i, 0, srcN-1))
				out[x].w = append(out[x].w, lanczos2Kernel(d))
			}
		}
		return out
	}
	xt := makeTaps(targetW, srcW, xscale, xf)
	yt := makeTaps(targetH, srcH, yscale, yf)
	srcRGBA, fast := src.(*image.RGBA)

	// Horizontal pass: srcW -> targetW, height unchanged.
	tmp := make([]float64, srcH*targetW*3)
	parallelRows(srcH, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			for x := 0; x < targetW; x++ {
				var wsum, rs, gs, bs float64
				for k, ii := range xt[x].idx {
					w := xt[x].w[k]
					var r, g, b uint32
					if fast {
						o := srcRGBA.PixOffset(srcB.Min.X+ii, srcB.Min.Y+y)
						r, g, b = uint32(srcRGBA.Pix[o]), uint32(srcRGBA.Pix[o+1]), uint32(srcRGBA.Pix[o+2])
					} else {
						r, g, b, _ = src.At(srcB.Min.X+ii, srcB.Min.Y+y).RGBA()
						r, g, b = r>>8, g>>8, b>>8
					}
					rs += w * float64(r)
					gs += w * float64(g)
					bs += w * float64(b)
					wsum += w
				}
				if wsum < 1e-9 {
					wsum = 1
				}
				o := (y*targetW + x) * 3
				tmp[o] = rs / wsum
				tmp[o+1] = gs / wsum
				tmp[o+2] = bs / wsum
			}
		}
	})

	// Vertical pass: srcH -> targetH.
	parallelRows(targetH, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			for x := 0; x < targetW; x++ {
				var wsum, rs, gs, bs float64
				for k, ii := range yt[y].idx {
					w := yt[y].w[k]
					o := (ii*targetW + x) * 3
					rs += w * tmp[o]
					gs += w * tmp[o+1]
					bs += w * tmp[o+2]
					wsum += w
				}
				if wsum < 1e-9 {
					wsum = 1
				}
				o := dst.PixOffset(x, y)
				dst.Pix[o] = uint8(clampF(rs/wsum, 0, 255))
				dst.Pix[o+1] = uint8(clampF(gs/wsum, 0, 255))
				dst.Pix[o+2] = uint8(clampF(bs/wsum, 0, 255))
				dst.Pix[o+3] = 255
			}
		}
	})
	return dst
}

// downsampleBox is the original box-average resize, kept as the gentle-
// scale path.
func downsampleBox(src image.Image, targetW, targetH int) *image.RGBA {
	srcB := src.Bounds()
	srcW := srcB.Dx()
	srcH := srcB.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	for y := 0; y < targetH; y++ {
		for x := 0; x < targetW; x++ {
			sx0 := srcB.Min.X + (x * srcW / targetW)
			sy0 := srcB.Min.Y + (y * srcH / targetH)
			sx1 := srcB.Min.X + ((x + 1) * srcW / targetW)
			sy1 := srcB.Min.Y + ((y + 1) * srcH / targetH)
			if sx1 <= sx0 {
				sx1 = sx0 + 1
			}
			if sy1 <= sy0 {
				sy1 = sy0 + 1
			}
			var rSum, gSum, bSum, count uint64
			for sy := sy0; sy < sy1; sy++ {
				for sx := sx0; sx < sx1; sx++ {
					r, g, b, _ := src.At(sx, sy).RGBA()
					rSum += uint64(r >> 8)
					gSum += uint64(g >> 8)
					bSum += uint64(b >> 8)
					count++
				}
			}
			if count == 0 {
				count = 1
			}
			dst.SetRGBA(x, y, color.RGBA{
				R: uint8(rSum / count),
				G: uint8(gSum / count),
				B: uint8(bSum / count),
				A: 255,
			})
		}
	}
	return dst
}

func compareImages(a, b *image.RGBA) float64 {
	bounds := a.Bounds()
	if b.Bounds().Dx() != bounds.Dx() || b.Bounds().Dy() != bounds.Dy() {
		return math.MaxFloat64
	}
	var sumSqDiff float64
	var count float64
	for y := 0; y < bounds.Dy(); y++ {
		for x := 0; x < bounds.Dx(); x++ {
			ar, ag, ab, _ := a.At(x, y).RGBA()
			br, bg, bb, _ := b.At(x, y).RGBA()
			dr := float64(ar>>8) - float64(br>>8)
			dg := float64(ag>>8) - float64(bg>>8)
			db := float64(ab>>8) - float64(bb>>8)
			sumSqDiff += dr*dr + dg*dg + db*db
			count += 3
		}
	}
	return sumSqDiff / count
}

// ============================================================================
// STRUCTURAL ANALYSIS (for reverse engineering)
// ============================================================================

// estimateExponent regresses log(power) vs log(freq) on the image's power
// spectrum — used both for initial estimation and structural scoring.
func estimateExponent(img *image.RGBA) float64 {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	padW := nextPow2(w)
	padH := nextPow2(h)
	grayData := make([][]complex128, padH)
	for y := 0; y < padH; y++ {
		grayData[y] = make([]complex128, padW)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			gray := float64(r+g+b) / 3.0 / 65535.0
			grayData[y][x] = complex(gray*2-1, 0)
		}
	}
	fft2d(grayData, false)
	halfW := padW / 2
	powerSum := make([]float64, halfW+1)
	powerCount := make([]int, halfW+1)
	for y := 0; y < padH; y++ {
		for x := 0; x < padW; x++ {
			fu := freqCoord(x, padW)
			fv := freqCoord(y, padH)
			r := int(math.Sqrt(fu*fu + fv*fv))
			if r >= 0 && r <= halfW {
				re := real(grayData[y][x])
				im := imag(grayData[y][x])
				powerSum[r] += re*re + im*im
				powerCount[r]++
			}
		}
	}
	var sumLF, sumLP, sumLF2, sumLFLP float64
	nPts := 0
	for r := 2; r < halfW; r++ {
		if powerCount[r] > 0 && powerSum[r] > 0 {
			avgP := powerSum[r] / float64(powerCount[r])
			lf := math.Log(float64(r))
			lp := math.Log(avgP)
			sumLF += lf
			sumLP += lp
			sumLF2 += lf * lf
			sumLFLP += lf * lp
			nPts++
		}
	}
	if nPts <= 2 {
		return 2.0
	}
	denom := float64(nPts)*sumLF2 - sumLF*sumLF
	if math.Abs(denom) < 1e-10 {
		return 2.0
	}
	slope := (float64(nPts)*sumLFLP - sumLF*sumLP) / denom
	return clampF(-slope, 0.5, 10.0)
}

func channelCorrelationOf(img *image.RGBA) float64 {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	var rVals, gVals, bVals []float64
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			rVals = append(rVals, float64(r>>8))
			gVals = append(gVals, float64(g>>8))
			bVals = append(bVals, float64(b>>8))
		}
	}
	return clampF((pearsonCorr(rVals, gVals)+pearsonCorr(rVals, bVals))/2, 0, 1)
}

// reverseScore measures how well a generated image matches the target's
// *statistical character*: spectral slope + channel correlation, mixed with
// raw MSE. Random-phase spectra can never match pixel-for-pixel, so pure MSE
// is a nearly hopeless objective — this finds genomes whose renders LOOK like
// the target even with a different seed. Returns (structural score, MSE);
// MSE is returned separately so the UI similarity % stays comparable.
func reverseScore(target, cand *image.RGBA) (score, mse float64) {
	mse = compareImages(target, cand)
	slopeDiff := math.Abs(estimateExponent(target) - estimateExponent(cand))
	corrDiff := math.Abs(channelCorrelationOf(target) - channelCorrelationOf(cand))
	return mse/(255.0*255.0) + 0.3*slopeDiff + 0.3*corrDiff, mse
}

func perturbGenome(g Genome, rng *rand.Rand) Genome {
	candidate := g
	step := rng.Float64()*2 - 1
	switch rng.Intn(17) {
	case 0:
		candidate.Exponent = clampF(candidate.Exponent+step*0.3, 0.5, 10.0)
	case 1:
		candidate.BandLimit = clampF(candidate.BandLimit+step*0.1, 0.01, 1.0)
	case 2:
		candidate.AxisStretch = clampF(candidate.AxisStretch+step*0.15, 0.25, 4.0)
	case 3:
		candidate.Gamma = clampF(candidate.Gamma+step*0.15, 0.3, 3.0)
	case 4:
		candidate.Colorfulness = clampF(candidate.Colorfulness+step*0.1, 0, 1)
	case 5:
		candidate.PalA = jitter3(candidate.PalA, 0.05, rng)
		candidate.PalB = jitter3(candidate.PalB, 0.05, rng)
	case 6:
		candidate.PalC = jitter3(candidate.PalC, 0.05, rng)
		candidate.PalD = jitter3(candidate.PalD, 0.05, rng)
	case 7:
		candidate.ExponentHi = clampF(candidate.ExponentHi+step*0.3, 0.5, 10.0)
		candidate.BreakFreq = clampF(candidate.BreakFreq+step*0.08, 0.0, 0.9)
	case 7 + 1:
		candidate.ReliefAngle += step * 0.4
		candidate.ReliefStrength = clampF(candidate.ReliefStrength+step*0.1, 0, 2)
	case 9:
		candidate.Transform = rng.Intn(4)
		if candidate.Transform == 2 || candidate.Transform == 3 {
			candidate.TerraceLevels = 3.0 + rng.Float64()*12.0
		}
	case 10:
		candidate.SpikeCount = 1 + rng.Intn(4)
		candidate.SpikeAmp = clampF(candidate.SpikeAmp+step*3.0, 0.5, 20.0)
	case 11:
		candidate.ChromaStrength = clampF(candidate.ChromaStrength+step*0.1, 0, 0.8)
	case 12:
		candidate.SpecRot += step * 0.8
		candidate.ConeAngle += step * 0.8
		if rng.Float64() < 0.3 {
			if candidate.ConeWidth > 0.999 {
				candidate.ConeWidth = 0.25 + rng.Float64()*0.5
			} else {
				candidate.ConeWidth = 1.0
			}
		}
	case 13:
		candidate.DomainWarp = clampF(candidate.DomainWarp+step*0.08, 0, 0.5)
	case 14:
		candidate.NormMode = rng.Intn(3)
	case 15:
		candidate.SymmetryFold = rng.Intn(9)
		if rng.Float64() < 0.4 {
			candidate.SymmetryMirror = !candidate.SymmetryMirror
		}
	case 16:
		candidate.PaletteMode = rng.Intn(2)
		if candidate.PaletteMode == 1 && candidate.AnchorCount < 2 {
			candidate.AnchorCount = 2 + rng.Intn(4)
			for k := 0; k < candidate.AnchorCount; k++ {
				for i := 0; i < 3; i++ {
					candidate.AnchorColors[k][i] = rng.Float64()
				}
			}
		}
	}
	if rng.Float64() < 0.1 {
		candidate.Seed = rng.Int63()
	}
	return candidate
}

// ============================================================================
// REVERSE ENGINEERING — METHOD 1: BRUTE FORCE
// ============================================================================

// ============================================================================
// REVERSE ENGINEERING v2 — DETERMINISTIC ANALYSIS & PHASE MATCH
// ============================================================================

type palPt struct{ t, v float64 }

// estimateAxisStretch compares spectral spread along fu vs fv.
// The synthesis stretches fv by s, so the ratio of the second moments
// recovers s directly.
func estimateAxisStretch(img *image.RGBA) float64 {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	padW, padH := nextPow2(w), nextPow2(h)
	data := make([][]complex128, padH)
	for y := 0; y < padH; y++ {
		data[y] = make([]complex128, padW)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			gray := float64(r+g+b) / 3.0 / 65535.0
			data[y][x] = complex(gray*2-1, 0)
		}
	}
	fft2d(data, false)
	var sw, swx2, swy2 float64
	for y := 0; y < padH; y++ {
		for x := 0; x < padW; x++ {
			fu := freqCoord(x, padW)
			fv := freqCoord(y, padH)
			p := real(data[y][x])*real(data[y][x]) + imag(data[y][x])*imag(data[y][x])
			sw += p
			swx2 += p * fu * fu
			swy2 += p * fv * fv
		}
	}
	if sw < 1e-12 {
		return 1.0
	}
	return clampF(math.Sqrt(swy2/sw)/math.Sqrt(swx2/sw), 0.25, 4.0)
}

// collectPalettePoints returns (normalized luminance, channel value) samples,
// sorted by t so the fit sees the palette as a coherent curve.
func collectPalettePoints(img *image.RGBA) (pts [3][]palPt) {
	b := img.Bounds()
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			r, g, b2, _ := img.At(x, y).RGBA()
			fr, fg, fb := float64(r>>8), float64(g>>8), float64(b2>>8)
			t := (fr + fg + fb) / 3.0 / 255.0 // same gray the renderer uses
			pts[0] = append(pts[0], palPt{t, fr / 255})
			pts[1] = append(pts[1], palPt{t, fg / 255})
			pts[2] = append(pts[2], palPt{t, fb / 255})
		}
	}
	for ch := 0; ch < 3; ch++ {
		sort.Slice(pts[ch], func(i, j int) bool { return pts[ch][i].t < pts[ch][j].t })
	}
	return
}

// fitChannel fits y ~= a + b*cos(2pi*(c*x + d)) by grid search over (c, d)
// with a closed-form linear solve for (a, b) at each grid point.
func fitChannel(pts []palPt, gamma float64) (a, b, c, d, bestErr float64) {
	n := len(pts)
	xs := make([]float64, n)
	for i, p := range pts {
		xs[i] = math.Pow(clampF(p.t, 0, 1), gamma)
	}
	eval := func(cc, dd float64) (aa, bb, e float64) {
		var s0, s1, s11, t0, t1 float64
		for i := range xs {
			c1 := math.Cos(2 * math.Pi * (cc*xs[i] + dd))
			s0++
			s1 += c1
			s11 += c1 * c1
			t0 += pts[i].v
			t1 += pts[i].v * c1
		}
		den := s0*s11 - s1*s1
		if math.Abs(den) < 1e-9 {
			return 0, 0, math.MaxFloat64
		}
		aa = (t0*s11 - t1*s1) / den
		bb = (s0*t1 - s1*t0) / den
		for i := range xs {
			pred := aa + bb*math.Cos(2*math.Pi*(cc*xs[i]+dd))
			dv := pts[i].v - pred
			e += dv * dv
		}
		return aa, bb, e
	}
	bestErr = math.MaxFloat64
	for ci := 0; ci <= 20; ci++ {
		cc := float64(ci) * 0.1
		for dj := 0; dj < 16; dj++ {
			dd := float64(dj) / 16.0
			aa, bb, e := eval(cc, dd)
			if e < bestErr {
				bestErr, a, b, c, d = e, aa, bb, cc, dd
			}
		}
	}
	c0, d0 := c, d
	for ci := -5; ci <= 5; ci++ {
		cc := c0 + float64(ci)*0.01
		for dj := -8; dj <= 8; dj++ {
			dd := math.Mod(d0+float64(dj)/128.0+2.0, 1.0)
			aa, bb, e := eval(cc, dd)
			if e < bestErr {
				bestErr, a, b, c, d = e, aa, bb, cc, dd
			}
		}
	}
	return
}

// fitPaletteGamma picks gamma and the full cosine palette jointly.
func fitPaletteGamma(pts [3][]palPt) (a, b, c, d [3]float64, gamma float64) {
	bestTotal := math.MaxFloat64
	store := func(pal [3][4]float64, g float64) {
		gamma = g
		for ch := 0; ch < 3; ch++ {
			a[ch], b[ch], c[ch], d[ch] = pal[ch][0], pal[ch][1], pal[ch][2], pal[ch][3]
		}
	}
	tryGamma := func(g float64) ([3][4]float64, float64) {
		var pal [3][4]float64
		total := 0.0
		for ch := 0; ch < 3; ch++ {
			pa, pb, pc, pd, e := fitChannel(pts[ch], g)
			pal[ch] = [4]float64{pa, pb, pc, pd}
			total += e
		}
		return pal, total
	}
	for gam := 0.3; gam <= 2.51; gam += 0.1 {
		pal, tot := tryGamma(gam)
		if tot < bestTotal {
			bestTotal = tot
			store(pal, gam)
		}
	}
	base := gamma
	for _, gam := range []float64{base - 0.08, base - 0.04, base + 0.04, base + 0.08} {
		if gam < 0.3 || gam > 2.5 {
			continue
		}
		pal, tot := tryGamma(gam)
		if tot < bestTotal {
			bestTotal = tot
			store(pal, gam)
		}
	}
	return
}

// analyzeGenome builds a genome from direct measurement of the target.
func analyzeGenome(target *image.RGBA) Genome {
	pts := collectPalettePoints(target)
	palA, palB, palC, palD, gamma := fitPaletteGamma(pts)
	return Genome{
		Seed:          0,
		Exponent:      estimateExponent(target),
		BandLimit:     0.6,
		AxisStretch:   estimateAxisStretch(target),
		Gamma:         gamma,
		Colorfulness:  channelCorrelationOf(target),
		MutationRate:  0,
		MutationPower: 0,
		PalA:          palA, PalB: palB, PalC: palC, PalD: palD,
	}
}

// fitRefSize caps a resolution so its longest side is maxSide,
// preserving aspect ratio.
func fitRefSize(w, h, maxSide int) (int, int) {
	long := w
	if h > long {
		long = h
	}
	if long <= maxSide {
		return w, h
	}
	s := float64(maxSide) / float64(long)
	return int(float64(w)*s + 0.5), int(float64(h)*s + 0.5)
}

// extractLumaRef encodes a downscaled copy of the target as a high-quality
// JPEG. JPEG keeps the reference embedded in the genome compact even at the
// large sizes needed for artifact-free full-resolution exports.
func extractLumaRef(target *image.RGBA, w, h int) (b64 string, lw, lh int) {
	src := downsampleImage(target, w, h)
	var buf bytes.Buffer
	jpeg.Encode(&buf, src, &jpeg.Options{Quality: 88})
	return base64.StdEncoding.EncodeToString(buf.Bytes()), w, h
}

// synthMatchChannel re-textures the stored luma map: keeps the target's FFT
// phase (spatial layout) and replaces amplitudes with the genome's spectral
// shaping. Output buffer is padded to power-of-two so generateSpectralImage's
// downstream indexing (y*padW + x) works unchanged for both paths.
func synthMatchChannel(width, height int, cfg Genome) []float64 {
	raw, err := base64.StdEncoding.DecodeString(cfg.LumaRef)
	if err != nil {
		return nil
	}
	lumaImg, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil
	}
	lw := cfg.LumaW
	lh := cfg.LumaH
	if lw != lumaImg.Bounds().Dx() || lh != lumaImg.Bounds().Dy() {
		lw = lumaImg.Bounds().Dx()
		lh = lumaImg.Bounds().Dy()
	}
	if lw < 2 || lh < 2 {
		return nil
	}

	// Preview/evolution renders happen at small canvas sizes. Downscale the
	// reference for those so the FFT work (and thus evolution speed) stays
	// cheap. Full-size exports keep the reference at native resolution and
	// therefore render with full detail instead of being upscaled.
	if width < lw/2 {
		scale := float64(width*2) / float64(lw)
		nh := int(float64(lh)*scale + 0.5)
		if nh < 2 {
			nh = 2
		}
		lumaImg = downsampleImage(lumaImg, width*2, nh)
		lw, lh = width*2, nh
	}

	padSrcW := nextPow2(lw)
	padSrcH := nextPow2(lh)

	data := make([][]complex128, padSrcH)
	for y := 0; y < padSrcH; y++ {
		data[y] = make([]complex128, padSrcW)
	}

	zoom := cfg.Zoom
	if zoom <= 0.01 {
		zoom = 1.0
	}
	cosR, sinR := math.Cos(-cfg.Rot), math.Sin(-cfg.Rot)
	s := math.Min(float64(lw), float64(lh))
	fcx := float64(lw)/2.0 + cfg.CenterX*float64(lw)
	fcy := float64(lh)/2.0 + cfg.CenterY*float64(lh)

	// Seed-driven liquid warp: one shared smooth field, deterministic from
	// Seed, amplitude = Warp (fraction of the frame). Identity at Warp = 0.
	warp := clampF(cfg.Warp, 0, 0.5)
	var hfx, hfy, hph [3]float64
	if warp > 0 {
		wr := rand.New(rand.NewSource(cfg.Seed + 7777))
		for h := 0; h < 3; h++ {
			hfx[h] = 1.5 + wr.Float64()*2.5
			hfy[h] = 1.5 + wr.Float64()*2.5
			hph[h] = wr.Float64() * 2 * math.Pi
		}
	}

	for y := 0; y < lh; y++ {
		ny := (float64(y) - float64(lh)/2.0) / s
		for x := 0; x < lw; x++ {
			nx := (float64(x) - float64(lw)/2.0) / s

			rx := (nx*cosR - ny*sinR) / zoom
			ry := (nx*sinR + ny*cosR) / zoom
			if cfg.FlipX > 0.5 {
				rx = -rx
			}
			if cfg.FlipY > 0.5 {
				ry = -ry
			}
			if warp > 0 {
				wd := 0.0
				for h := 0; h < 3; h++ {
					wd += math.Sin(2*math.Pi*(hfx[h]*rx+hfy[h]*ry) + hph[h])
				}
				off := warp * wd / 3.0
				rx += off
				ry += off * 0.8
			}

			sxi := int(rx*s + fcx + 0.5)
			syi := int(ry*s + fcy + 0.5)
			sxi = ((sxi % lw) + lw) % lw
			syi = ((syi % lh) + lh) % lh

			r, g, b, _ := lumaImg.At(sxi, syi).RGBA()
			gray := (float64(r>>8) + float64(g>>8) + float64(b>>8)) / 3.0 / 255.0
			data[y][x] = complex(gray*2-1, 0)
		}
	}
	fft2d(data, false)

	mix := clampF(cfg.PhaseMix, 0, 1)
	var jitRng *rand.Rand
	if cfg.PhaseJitter > 0 {
		jitRng = rand.New(rand.NewSource(cfg.Seed))
	}
	for y := 0; y < padSrcH; y++ {
		for x := 0; x < padSrcW; x++ {
			re, im := real(data[y][x]), imag(data[y][x])
			origMag := math.Hypot(re, im)
			if origMag < 1e-12 {
				continue
			}
			fu := freqCoord(x, padSrcW)
			fv := freqCoord(y, padSrcH)
			fvv := fv * cfg.AxisStretch
			f := math.Sqrt(fu*fu + fvv*fvv)

			targetAmp := origMag
			if f >= 0.5 {
				desired := 1.0 / math.Pow(f, cfg.Exponent/2.0)
				targetAmp = (origMag + desired) / 2.0
			}
			if cfg.BandLimit > 0 {
				cutoff := float64(padSrcW/2) * cfg.BandLimit
				if f > cutoff {
					targetAmp *= math.Exp(-((f - cutoff) * (f - cutoff)) / (2 * cutoff * cutoff))
				}
			}
			ph := math.Atan2(im, re) * mix
			if jitRng != nil {
				ph += (jitRng.Float64()*2 - 1) * cfg.PhaseJitter
			}
			data[y][x] = complex(targetAmp*math.Cos(ph), targetAmp*math.Sin(ph))
		}
	}
	fft2d(data, true)

	padOutW := nextPow2(width)
	padOutH := nextPow2(height)
	out := make([]float64, padOutW*padOutH)
	// 1-pixel-wide/high canvases would divide by zero (NaN indices -> panic).
	denW, denH := float64(maxInt(width-1, 1)), float64(maxInt(height-1, 1))
	for y := 0; y < height; y++ {
		fy := float64(y) * float64(lh-1) / denH
		y0 := int(fy)
		dy := fy - float64(y0)
		y1 := minInt(y0+1, lh-1)
		for x := 0; x < width; x++ {
			fx := float64(x) * float64(lw-1) / denW
			x0 := int(fx)
			dx := fx - float64(x0)
			x1 := minInt(x0+1, lw-1)
			top := real(data[y0][x0])*(1-dx) + real(data[y0][x1])*dx
			bot := real(data[y1][x0])*(1-dx) + real(data[y1][x1])*dx
			out[y*padOutW+x] = top*(1-dy) + dy*bot
		}
	}
	return out
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func reverseEngineerPrecise(target *image.RGBA, refineIter int) (Genome, float64) {
	const cmpW, cmpH = 128, 96
	cmpTarget := downsampleImage(target, cmpW, cmpH)
	tgtExp := estimateExponent(cmpTarget)
	tgtCorr := channelCorrelationOf(cmpTarget)
	scoreWith := func(cand *image.RGBA) (float64, float64) {
		mse := compareImages(cmpTarget, cand)
		sd := math.Abs(estimateExponent(cand) - tgtExp)
		cd := math.Abs(channelCorrelationOf(cand) - tgtCorr)
		return mse/(255.0*255.0) + 0.3*sd + 0.3*cd, mse
	}

	genome := analyzeGenome(cmpTarget)

	// Probe several seeds: with random phases only the texture character can
	// match, so pick the luckiest realization.
	searchRng := rand.New(rand.NewSource(time.Now().UnixNano()))
	bestScore := math.MaxFloat64
	bestMSE := math.MaxFloat64
	for s := 0; s < 8; s++ {
		g := genome
		g.Seed = searchRng.Int63()
		genImg := generateSpectralImage(cmpW, cmpH, rand.New(rand.NewSource(g.Seed)), g)
		sc, mse := scoreWith(genImg)
		if sc < bestScore {
			bestScore, bestMSE, genome = sc, mse, g
		}
	}
	// Short hill climb to polish.
	for iter := 0; iter < refineIter; iter++ {
		cand := perturbGenome(genome, searchRng)
		genImg := generateSpectralImage(cmpW, cmpH, rand.New(rand.NewSource(cand.Seed)), cand)
		sc, mse := scoreWith(genImg)
		if sc < bestScore {
			bestScore, bestMSE, genome = sc, mse, cand
		}
	}
	return genome, bestMSE
}

func reverseEngineerMatch(target image.Image, refineIter int) (Genome, float64) {
	const cmpW, cmpH = 128, 96
	cmpTarget := downsampleImage(target, cmpW, cmpH)

	genome := analyzeGenome(cmpTarget)
	// Store a high-resolution luma reference (capped at 1920 px on the long
	// side) so exports render natively at the requested size instead of
	// being upscaled from a small map.
	refW, refH := fitRefSize(target.Bounds().Dx(), target.Bounds().Dy(), 1920)
	if refW < 2 || refH < 2 {
		refW, refH = cmpW, cmpH
	}
	lumaB64, lw, lh := extractLumaRef(downsampleImage(target, refW, refH), refW, refH)
	genome.LumaRef, genome.LumaW, genome.LumaH = lumaB64, lw, lh
	genome.PhaseMix = 1.0
	genome.Structure = 1.0

	tgtExp := estimateExponent(cmpTarget)
	tgtCorr := channelCorrelationOf(cmpTarget)
	scoreOf := func(cand Genome) (float64, float64) {
		img := generateSpectralImage(cmpW, cmpH, rand.New(rand.NewSource(cand.Seed)), cand)
		mse := compareImages(cmpTarget, img)
		sd := math.Abs(estimateExponent(img) - tgtExp)
		cd := math.Abs(channelCorrelationOf(img) - tgtCorr)
		return mse/(255.0*255.0) + 0.3*sd + 0.3*cd, mse
	}

	bestScore, bestMSE := scoreOf(genome)
	searchRng := rand.New(rand.NewSource(time.Now().UnixNano()))
	for iter := 0; iter < refineIter; iter++ {
		cand := perturbGenome(genome, searchRng)
		// Never let the search drop the phase reference.
		cand.LumaRef, cand.LumaW, cand.LumaH = genome.LumaRef, genome.LumaW, genome.LumaH
		cand.PhaseMix = genome.PhaseMix
		if sc, mse := scoreOf(cand); sc < bestScore {
			bestScore, bestMSE, genome = sc, mse, cand
		}
	}
	// Return the MSE like every other method: the handler converts it to a
	// similarity percentage, and the structural score (~0..1) made "match"
	// always report ~100%.
	return genome, bestMSE
}

func reverseEngineerBruteForce(target *image.RGBA, iterations int) (Genome, float64) {
	const cmpW, cmpH = 64, 48
	cmpTarget := downsampleImage(target, cmpW, cmpH)
	// Cached target statistics (were recomputed on every iteration).
	tgtExp := estimateExponent(cmpTarget)
	tgtCorr := channelCorrelationOf(cmpTarget)
	scoreWith := func(cand *image.RGBA) (float64, float64) {
		mse := compareImages(cmpTarget, cand)
		sd := math.Abs(estimateExponent(cand) - tgtExp)
		cd := math.Abs(channelCorrelationOf(cand) - tgtCorr)
		return mse/(255.0*255.0) + 0.3*sd + 0.3*cd, mse
	}
	bestGenome := Genome{}
	bestScore := math.MaxFloat64
	bestMSE := math.MaxFloat64
	for i := 0; i < iterations; i++ {
		seedRng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(i)*999999))
		genome := randomGenome(seedRng)
		imgRng := rand.New(rand.NewSource(genome.Seed))
		genImg := generateSpectralImage(cmpW, cmpH, imgRng, genome)
		score, mse := scoreWith(genImg)
		if score < bestScore {
			bestScore = score
			bestGenome = genome
			bestMSE = mse
		}
	}
	return bestGenome, bestMSE
}

// ============================================================================
// REVERSE ENGINEERING — METHOD 2: HILL CLIMBING
// ============================================================================

func reverseEngineerHillClimb(target *image.RGBA, iterations int) (Genome, float64) {
	const cmpW, cmpH = 64, 48
	cmpTarget := downsampleImage(target, cmpW, cmpH)
	tgtExp := estimateExponent(cmpTarget)
	tgtCorr := channelCorrelationOf(cmpTarget)
	scoreWith := func(cand *image.RGBA) (float64, float64) {
		mse := compareImages(cmpTarget, cand)
		sd := math.Abs(estimateExponent(cand) - tgtExp)
		cd := math.Abs(channelCorrelationOf(cand) - tgtCorr)
		return mse/(255.0*255.0) + 0.3*sd + 0.3*cd, mse
	}
	searchRng := rand.New(rand.NewSource(time.Now().UnixNano()))
	genome := randomGenome(searchRng)
	imgRng := rand.New(rand.NewSource(genome.Seed))
	genImg := generateSpectralImage(cmpW, cmpH, imgRng, genome)
	bestScore, bestMSE := scoreWith(genImg)
	noImproveCount := 0
	for iter := 0; iter < iterations; iter++ {
		candidate := perturbGenome(genome, searchRng)
		imgRng := rand.New(rand.NewSource(candidate.Seed))
		genImg := generateSpectralImage(cmpW, cmpH, imgRng, candidate)
		score, mse := scoreWith(genImg)
		if score < bestScore {
			bestScore = score
			bestMSE = mse
			genome = candidate
			noImproveCount = 0
		} else {
			noImproveCount++
			if noImproveCount > 100 {
				genome = randomGenome(searchRng)
				imgRng := rand.New(rand.NewSource(genome.Seed))
				genImg := generateSpectralImage(cmpW, cmpH, imgRng, genome)
				bestScore, bestMSE = scoreWith(genImg)
				noImproveCount = 0
			}
		}
	}
	return genome, bestMSE
}

// ============================================================================
// REVERSE ENGINEERING — METHOD 3: ESTIMATE + FINE-TUNE
// ============================================================================

func reverseEngineerEstimate(target *image.RGBA, fineTuneIter int) (Genome, float64) {
	const cmpW, cmpH = 64, 48
	cmpTarget := downsampleImage(target, cmpW, cmpH)
	tgtExp := estimateExponent(cmpTarget)
	tgtCorr := channelCorrelationOf(cmpTarget)
	scoreWith := func(cand *image.RGBA) (float64, float64) {
		mse := compareImages(cmpTarget, cand)
		sd := math.Abs(estimateExponent(cand) - tgtExp)
		cd := math.Abs(channelCorrelationOf(cand) - tgtCorr)
		return mse/(255.0*255.0) + 0.3*sd + 0.3*cd, mse
	}

	genome := Genome{
		Seed:          time.Now().UnixNano(),
		Exponent:      tgtExp,
		BandLimit:     0.5,
		AxisStretch:   1.0,
		Gamma:         1.0,
		Colorfulness:  tgtCorr,
		MutationRate:  0.0,
		MutationPower: 0.0,
		// Neutral grayscale ramp (0.5 - 0.5*cos(pi*t)) as the starting
		// palette. Left zeroed, the cosine palette is solid black, so the
		// search started from (and mostly stayed at) a near-black image.
		PalA: [3]float64{0.5, 0.5, 0.5},
		PalB: [3]float64{0.5, 0.5, 0.5},
		PalC: [3]float64{0.5, 0.5, 0.5},
		PalD: [3]float64{0.5, 0.5, 0.5},
	}

	searchRng := rand.New(rand.NewSource(time.Now().UnixNano()))
	imgRng := rand.New(rand.NewSource(genome.Seed))
	genImg := generateSpectralImage(cmpW, cmpH, imgRng, genome)
	bestScore, bestMSE := scoreWith(genImg)
	noImproveCount := 0
	for iter := 0; iter < fineTuneIter; iter++ {
		candidate := perturbGenome(genome, searchRng)
		imgRng := rand.New(rand.NewSource(candidate.Seed))
		genImg := generateSpectralImage(cmpW, cmpH, imgRng, candidate)
		score, mse := scoreWith(genImg)
		if score < bestScore {
			bestScore = score
			bestMSE = mse
			genome = candidate
			noImproveCount = 0
		} else {
			noImproveCount++
			if noImproveCount > 80 {
				genome = randomGenome(searchRng)
				imgRng := rand.New(rand.NewSource(genome.Seed))
				genImg := generateSpectralImage(cmpW, cmpH, imgRng, genome)
				bestScore, bestMSE = scoreWith(genImg)
				noImproveCount = 0
			}
		}
	}
	return genome, bestMSE
}

func pearsonCorr(a, b []float64) float64 {
	n := float64(len(a))
	if n == 0 {
		return 0
	}
	var sa, sb float64
	for i := range a {
		sa += a[i]
		sb += b[i]
	}
	ma := sa / n
	mb := sb / n
	var cov, va, vb float64
	for i := range a {
		da := a[i] - ma
		db := b[i] - mb
		cov += da * db
		va += da * da
		vb += db * db
	}
	if va < 1e-10 || vb < 1e-10 {
		return 0
	}
	return cov / math.Sqrt(va*vb)
}

// ============================================================================
// EASING FUNCTIONS
// ============================================================================

// easeLinear - No easing, constant speed
func easeLinear(t float64) float64 { return t }

func easeInQuad(t float64) float64 { return t * t }

func easeOutQuad(t float64) float64 { return t * (2.0 - t) }

func easeInOutQuad(t float64) float64 {
	if t < 0.5 {
		return 2.0 * t * t
	}
	return 1.0 - math.Pow(-2.0*t+2.0, 2.0)/2.0
}

func easeInCubic(t float64) float64 { return t * t * t }

func easeOutCubic(t float64) float64 { return math.Pow(t-1.0, 3.0) + 1.0 }

func easeInOutCubic(t float64) float64 {
	// Canonical inout-cubic: 4t³ below the midpoint, 1 - (−2t+2)³/2 above.
	// The old second branch leaked the Back-easing constant (numerator +4
	// instead of +1), so the curve jumped to 1.5 exactly at t = 0.5 —
	// discontinuous, saturating the second half of any animation.
	if t < 0.5 {
		return math.Pow(2.0*t, 3.0) / 2.0
	}
	return 1.0 + math.Pow(2.0*t-2.0, 3.0)/2.0
}

func easeInQuart(t float64) float64 { return t * t * t * t }

func easeInOutQuart(t float64) float64 {
	if t < 0.5 {
		return 8.0 * t * t * t * t
	}
	return 1.0 - math.Pow(-2.0*t+2.0, 4.0)/2.0
}

func easeOutQuart(t float64) float64 {
	return 1.0 - math.Pow(1.0-t, 4.0)
}

func easeInQuint(t float64) float64 { return t * t * t * t * t }

func easeOutQuint(t float64) float64 { return math.Pow(t-1.0, 5.0) + 1.0 }

func easeInOutQuint(t float64) float64 {
	// Canonical form: 16t⁴ … 1 + (2t−2)⁵/2. The old body returned
	// (pow(2t−2, 5) + 1)/2, landing at 0.5 at t = 1 — the morph visibly
	// ended half-done. Same defect class as easeInOutCubic.
	if t < 0.5 {
		return math.Pow(2.0*t, 5.0) / 2.0
	}
	return 1.0 + math.Pow(2.0*t-2.0, 5.0)/2.0
}

func easeSin(t float64) float64 { return (1.0 - math.Cos(t*math.Pi)) / 2.0 }

func easeInSin(t float64) float64 { return 1.0 - math.Cos(t*math.Pi/2.0) }

func easeOutSin(t float64) float64 { return math.Sin(t * math.Pi / 2.0) }

func easeInOutSin(t float64) float64 { return -(math.Cos(math.Pi*t) - 1.0) / 2.0 }

func easeExpoIn(t float64) float64 {
	if t == 0 {
		return 0
	}
	return math.Pow(2.0, 10.0*t-10.0)
}

func easeExpoOut(t float64) float64 {
	if t == 1 {
		return 1
	}
	return 1.0 - math.Pow(2.0, -10.0*t)
}

func easeExpoInOut(t float64) float64 {
	if t == 0 {
		return 0
	}
	if t == 1 {
		return 1
	}
	if t < 0.5 {
		return math.Pow(2.0, 20.0*t-10.0) / 2.0
	}
	return (2.0 - math.Pow(2.0, -20.0*t+10.0)) / 2.0
}

func easeCircleIn(t float64) float64 { return 1.0 - math.Sqrt(1.0-t*t) }

func easeCircleOut(t float64) float64 { return math.Sqrt(1.0 - (t-1.0)*(t-1.0)) }

func easeCircleInOut(t float64) float64 {
	if t < 0.5 {
		return (1.0 - math.Sqrt(1.0-4.0*(t*0.5)*(t*0.5))) / 2.0
	}
	return (math.Sqrt(1.0-(-2.0*t+2.0)*(-2.0*t+2.0)) + 1.0) / 2.0
}

func easeBackIn(t float64) float64 {
	c1 := 1.70158
	c3 := c1 + 1.0
	return c3*t*t*t - c1*t*t
}

func easeBackOut(t float64) float64 {
	c1 := 1.70158
	c3 := c1 + 1.0
	return 1.0 + c3*math.Pow(t-1.0, 3.0) + c1*math.Pow(t-1.0, 2.0)
}

func easeBackInOut(t float64) float64 {
	c1 := 1.70158
	c2 := c1 * 1.525
	if t < 0.5 {
		return (t * 2.0) * (t * 2.0) * ((c2+1.0)*t*2.0 - c2) / 2.0
	}
	return ((t*2.0-2.0)*(t*2.0-2.0)*((c2+1.0)*(t*2.0-2.0)+c2) + 2.0) / 2.0
}

func easeElasticIn(t float64) float64 {
	if t == 0 || t == 1 {
		return t
	}
	c4 := (2.0 * math.Pi) / 3.0
	return -math.Pow(2.0, 10.0*t-10.0) * math.Sin((t*10.0-10.75)*c4)
}

func easeElasticOut(t float64) float64 {
	if t == 0 || t == 1 {
		return t
	}
	c4 := (2.0 * math.Pi) / 3.0
	return math.Pow(2.0, -10.0*t)*math.Sin((t*10.0-0.75)*c4) + 1.0
}

func easeElasticInOut(t float64) float64 {
	if t == 0 {
		return 0
	}
	if t == 1 {
		return 1
	}
	c4 := (2.0 * math.Pi) / 3.0
	if t < 0.5 {
		return -(math.Pow(2.0, 20.0*t-10.0) * math.Sin((t*20.0-11.125)*c4)) / 2.0
	}
	return (math.Pow(2.0, -20.0*t+10.0)*math.Sin((t*20.0-11.125)*c4))/2.0 + 1.0
}

func easeBounceOut(t float64) float64 {
	n1 := 7.5625
	d1 := 2.75
	if t < 1.0/d1 {
		return n1 * t * t
	} else if t < 2.0/d1 {
		t -= 1.5 / d1
		return n1*t*t + 0.75
	} else if t < 2.5/d1 {
		t -= 2.25 / d1
		return n1*t*t + 0.9375
	}
	t -= 2.625 / d1
	return n1*t*t + 0.984375
}

func easeBounceIn(t float64) float64 { return 1.0 - easeBounceOut(1.0-t) }

func easeBounceInOut(t float64) float64 {
	if t < 0.5 {
		return (1.0 - easeBounceOut(1.0-2.0*t)) / 2.0
	}
	return (1.0 + easeBounceOut(2.0*t-1.0)) / 2.0
}

// easeGauss is the MONOTONE Gaussian-CDF ease: the integral of a bell
// centered at t = 0.5. Unlike the bell itself (which peaked at 1.0
// mid-timeline and retracted, producing A -> B -> A palindromes), the
// CDF runs strictly 0 -> 1, so it is a legitimate start-to-end easing
// with a pronounced slow-fast-slow character.
func easeGauss(t float64) float64 {
	sigma := 0.25
	scale := 0.5 / (sigma * math.Sqrt2) // = 1/erf-argument at the ends
	// numerator: erf shifted so t=0 maps to the curve's left tail
	num := math.Erf((t-0.5)/(sigma*math.Sqrt2)) + math.Erf(scale)
	den := 2.0 * math.Erf(scale)
	if den < 1e-12 {
		return t
	}
	return num / den
}

func getEasingFunc(name string) func(float64) float64 {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "linear":
		return easeLinear
	case "inquad", "easeinquad":
		return easeInQuad
	case "outquad", "easeoutquad":
		return easeOutQuad
	case "inoutquad", "easeinoutquad":
		return easeInOutQuad
	case "incubic", "easeincubic":
		return easeInCubic
	case "outcubic", "easeoutcubic":
		return easeOutCubic
	case "inoutcubic", "easeinoutcubic":
		return easeInOutCubic
	case "inquart", "easeinquart":
		return easeInQuart
	case "outquart", "easeoutquart":
		return easeOutQuart
	case "inoutquart", "easeinoutquart":
		return easeInOutQuart
	case "inquint", "easeinquint":
		return easeInQuint
	case "outquint", "easeoutquint":
		return easeOutQuint
	case "inoutquint", "easeinoutquint":
		return easeInOutQuint
	case "sin", "sine", "ease":
		return easeSin
	case "insin", "easeinsin":
		return easeInSin
	case "outsin", "easeoutsin":
		return easeOutSin
	case "inoutsin", "easeinoutsin":
		return easeInOutSin
	case "expo":
		return easeExpoInOut
	case "expoin", "easeexpon":
		return easeExpoIn
	case "expoout", "easeexpoout":
		return easeExpoOut
	case "circle", "circ":
		return easeCircleInOut
	case "circlein", "circin":
		return easeCircleIn
	case "circleout", "circout":
		return easeCircleOut
	case "back":
		return easeBackInOut
	case "backin":
		return easeBackIn
	case "backout":
		return easeBackOut
	case "elastic":
		return easeElasticInOut
	case "elasticin":
		return easeElasticIn
	case "elasticout":
		return easeElasticOut
	case "bounce":
		return easeBounceInOut
	case "bouncein":
		return easeBounceIn
	case "bounceout":
		return easeBounceOut
	case "gauss", "gaussian":
		return easeGauss
	default:
		return easeLinear
	}
}

// ============================================================================
// ANIMATION RENDERER
// ============================================================================

func lerp3(a, b [3]float64, t float64) [3]float64 {
	var out [3]float64
	for i := 0; i < 3; i++ {
		out[i] = a[i] + (b[i]-a[i])*t
	}
	return out
}

// lerpAnchors blends anchor color sets; only A's count is rendered, so
// blending the full array is safe for animation.
func lerpAnchors(a, b [5][3]float64, t float64) [5][3]float64 {
	var out [5][3]float64
	for k := 0; k < 5; k++ {
		out[k] = lerp3(a[k], b[k], t)
	}
	return out
}

// lerpAngle interpolates angles along the SHORTEST arc so genes like
// Rot/SpecRot/ConeAngle never spin the long way around during a morph
// (A=-3.13, B=+3.13 must be a 0.03-rad step, not a full 2*pi sweep).
func lerpAngle(a, b, t float64) float64 {
	d := math.Mod(b-a+3*math.Pi, 2*math.Pi) - math.Pi
	return a + d*t
}

// lerpPhase lerps phase components in [0,1) wrapping through zero, so a
// palette phase going 0.95 -> 0.05 takes the short path instead of
// sweeping through the whole palette mid-morph.
func lerpPhase(a, b, t float64) float64 {
	d := math.Mod(b-a+1.5, 1.0) - 0.5
	return math.Mod(a+d*t+1.0, 1.0)
}

// interpolateGenomes builds the frame genome between A (progress 0) and
// B (progress 1). Fixes:
//
//	Bug 5: starts from a full COPY of A, so LumaRef/structure genes survive
//	       even when the two cells carry different references (the old
//	       struct literal silently zeroed them for every frame).
//	Bug 2: every lerped scalar is clamped to its valid range, so overshoot
//	       easings can no longer push genes out of range.
//	Bug 7: angular genes and palette phases take the shortest path.
//	Bug 6 (partially): Seed and discrete genes (Transform, SpikeCount,
//	       NormMode, SymmetryFold, PaletteMode, AnchorCount, Flips) are kept
//	       from A for every frame. They cannot be meaningfully lerped; the
//	       animation handler bridges them with a late crossfade to the true
//	       render of B (fix 4) so the morph always ENDS at cell B.
func interpolateGenomes(genomeA, genomeB Genome, progress float64) Genome {
	out := genomeA

	// lerp-with-clamp for scalar genes.
	l := func(a, b, lo, hi float64) float64 {
		return clampF(a+(b-a)*progress, lo, hi)
	}

	out.Exponent = l(genomeA.Exponent, genomeB.Exponent, 0.5, 10.0)
	out.BandLimit = l(genomeA.BandLimit, genomeB.BandLimit, 0.01, 1.0)
	out.AxisStretch = l(genomeA.AxisStretch, genomeB.AxisStretch, 0.25, 4.0)
	out.Gamma = l(genomeA.Gamma, genomeB.Gamma, 0.3, 3.0)
	out.Colorfulness = l(genomeA.Colorfulness, genomeB.Colorfulness, 0.0, 1.0)
	out.MutationRate = l(genomeA.MutationRate, genomeB.MutationRate, 0.0001, 0.1)
	out.MutationPower = l(genomeA.MutationPower, genomeB.MutationPower, 1.0, 100.0)
	out.TerraceLevels = l(genomeA.TerraceLevels, genomeB.TerraceLevels, 2.0, 24.0)
	out.ReliefStrength = l(genomeA.ReliefStrength, genomeB.ReliefStrength, 0.0, 2.0)
	out.ExponentHi = l(genomeA.ExponentHi, genomeB.ExponentHi, 0.5, 10.0)
	out.BreakFreq = l(genomeA.BreakFreq, genomeB.BreakFreq, 0.0, 0.9)
	out.SpikeAmp = l(genomeA.SpikeAmp, genomeB.SpikeAmp, 0.0, 20.0)
	out.ChromaStrength = l(genomeA.ChromaStrength, genomeB.ChromaStrength, 0.0, 0.8)
	out.ConeWidth = l(genomeA.ConeWidth, genomeB.ConeWidth, 0.0, 1.0)
	out.DomainWarp = l(genomeA.DomainWarp, genomeB.DomainWarp, 0.0, 0.5)
	out.WarpScale = l(genomeA.WarpScale, genomeB.WarpScale, 0.0, 10.0)
	// Layer: continuous genes lerp only when both cells are layered (Mode
	// itself stays A's and is bridged by the end-of-segment crossfade).
	// Cellular: jitter and mix lerp when both cells are cellular; points
	// move via the phase morph; mode, metric and cell count stay A's and
	// are bridged by the end-of-segment crossfade.
	if genomeA.Lic.Mode > 0 && genomeB.Lic.Mode > 0 {
		out.Lic.Length = l(genomeA.Lic.Length, genomeB.Lic.Length, 1, 60)
		out.Lic.Mix = l(genomeA.Lic.Mix, genomeB.Lic.Mix, 0, 1)
		out.Lic.FlowScale = l(genomeA.Lic.FlowScale, genomeB.Lic.FlowScale, 0, 10)
		out.Lic.Smooth = l(genomeA.Lic.Smooth, genomeB.Lic.Smooth, 1, 40)
	}
	if genomeA.Cell.Mode > 0 && genomeB.Cell.Mode > 0 {
		out.Cell.Jitter = l(genomeA.Cell.Jitter, genomeB.Cell.Jitter, 0, 1)
		out.Cell.Mix = l(genomeA.Cell.Mix, genomeB.Cell.Mix, 0, 1)
	}
	if genomeA.Layer.Mode > 0 && genomeB.Layer.Mode > 0 {
		la, lb := genomeA.Layer, genomeB.Layer
		out.Layer.Exponent = l(la.Exponent, lb.Exponent, 0.5, 10)
		out.Layer.ExponentHi = l(la.ExponentHi, lb.ExponentHi, 0.5, 10)
		out.Layer.BreakFreq = l(la.BreakFreq, lb.BreakFreq, 0, 0.9)
		out.Layer.BandLimit = l(la.BandLimit, lb.BandLimit, 0.01, 1)
		out.Layer.AxisStretch = l(la.AxisStretch, lb.AxisStretch, 0.25, 4)
		out.Layer.SpecRot = lerpAngle(la.SpecRot, lb.SpecRot, progress)
		out.Layer.ConeAngle = lerpAngle(la.ConeAngle, lb.ConeAngle, progress)
		out.Layer.ConeWidth = l(la.ConeWidth, lb.ConeWidth, 0, 1)
		out.Layer.Mix = l(la.Mix, lb.Mix, 0, 1)
		out.Layer.MaskScale = l(la.MaskScale, lb.MaskScale, 0, 8)
		out.Layer.MaskSharp = l(la.MaskSharp, lb.MaskSharp, 0, 40)
		out.Layer.MaskBias = l(la.MaskBias, lb.MaskBias, -1, 1)
	}

	// Angular genes: shortest-arc interpolation, result wrapped to [0, 2*pi).
	out.ReliefAngle = math.Mod(lerpAngle(genomeA.ReliefAngle, genomeB.ReliefAngle, progress)+2*math.Pi, 2*math.Pi)
	out.SpecRot = math.Mod(lerpAngle(genomeA.SpecRot, genomeB.SpecRot, progress)+2*math.Pi, 2*math.Pi)
	out.ConeAngle = math.Mod(lerpAngle(genomeA.ConeAngle, genomeB.ConeAngle, progress)+2*math.Pi, 2*math.Pi)

	// Cosine palette: vectors lerp and clamp (the renderer tolerates the
	// full range, this is belt-and-braces), phases wrap.
	for ch := 0; ch < 3; ch++ {
		out.PalA[ch] = clampF(genomeA.PalA[ch]+(genomeB.PalA[ch]-genomeA.PalA[ch])*progress, 0.0, 2.0)
		out.PalB[ch] = clampF(genomeA.PalB[ch]+(genomeB.PalB[ch]-genomeA.PalB[ch])*progress, 0.0, 2.0)
		out.PalC[ch] = clampF(genomeA.PalC[ch]+(genomeB.PalC[ch]-genomeA.PalC[ch])*progress, 0.0, 2.0)
		out.PalD[ch] = lerpPhase(genomeA.PalD[ch], genomeB.PalD[ch], progress)
	}

	// Anchor palette: anchor count stays at A's (discrete), colors lerp.
	out.AnchorColors = lerpAnchors(genomeA.AnchorColors, genomeB.AnchorColors, progress)

	// OKLCH palette: magnitudes lerp, phases and base hue wrap the short way.
	la, lb := genomeA.Lch, genomeB.Lch
	lp := func(a, b float64) float64 { return a + (b-a)*progress }
	out.Lch = LchPalette{
		L0: lp(la.L0, lb.L0), LAmp: lp(la.LAmp, lb.LAmp), LFreq: lp(la.LFreq, lb.LFreq),
		LPhase: lerpPhase(la.LPhase, lb.LPhase, progress),
		C0:     lp(la.C0, lb.C0), CAmp: lp(la.CAmp, lb.CAmp), CFreq: lp(la.CFreq, lb.CFreq),
		CPhase: lerpPhase(la.CPhase, lb.CPhase, progress),
		H0:     lerpPhase(la.H0, lb.H0, progress), HSpan: lp(la.HSpan, lb.HSpan),
	}

	// Match-mode structure genes. The copy above already guarantees
	// reasonable values when the references differ; lerp them only when
	// they describe the SAME reference, otherwise two different photos
	// would be averaged geometrically.
	if out.LumaRef != "" && genomeA.LumaRef == genomeB.LumaRef {
		out.PhaseMix = l(genomeA.PhaseMix, genomeB.PhaseMix, 0.0, 1.0)
		out.PhaseJitter = l(genomeA.PhaseJitter, genomeB.PhaseJitter, 0.0, 1.2)
		out.Zoom = l(genomeA.Zoom, genomeB.Zoom, 0.9, 2.2)
		out.Rot = math.Mod(lerpAngle(genomeA.Rot, genomeB.Rot, progress)+3*math.Pi, 2*math.Pi) - math.Pi
		out.CenterX = l(genomeA.CenterX, genomeB.CenterX, -0.25, 0.25)
		out.CenterY = l(genomeA.CenterY, genomeB.CenterY, -0.25, 0.25)
		out.Structure = l(genomeA.Structure, genomeB.Structure, 0.0, 1.0)
		out.Warp = l(genomeA.Warp, genomeB.Warp, 0.0, 0.5)
	}

	// Deliberately kept from A (see doc comment): Seed, Transform,
	// SpikeCount, NormMode, SymmetryFold/Mirror, PaletteMode, AnchorCount,
	// FlipX/FlipY. The handler bridges these via crossfade (fix 4).
	return out
}

// blendImages performs a linear pixel crossfade: result = A*(1-t) + B*t
//
// Bug 2 fix: t is clamped to [0, 1] at the top. Easings like Back and
// Elastic deliberately overshoot (t < 0 or t > 1), and the old code fed
// those weights straight into a uint8 conversion. In Go, converting a
// float32 that is negative or > 255 to uint8 is implementation-defined —
// you get essentially arbitrary byte values, which is the garbage-pixels
// flicker seen with overshoot easings in crossfade mode.
// clamped, ta is always in [0,1] and every weighted sum lands in [0,255].
func blendImages(a, b *image.RGBA, t float64) *image.RGBA {
	bounds := a.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	out := image.NewRGBA(image.Rect(0, 0, w, h))

	tb := float32(clampF(t, 0.0, 1.0))
	ta := float32(1.0) - tb

	ap := a.Pix
	bp := b.Pix
	op := out.Pix

	for i := 0; i < len(ap); i += 4 {
		op[i] = uint8(float32(ap[i])*ta + float32(bp[i])*tb)
		op[i+1] = uint8(float32(ap[i+1])*ta + float32(bp[i+1])*tb)
		op[i+2] = uint8(float32(ap[i+2])*ta + float32(bp[i+2])*tb)
		op[i+3] = 255
	}
	return out
}

// cleanupPartialAnim removes a failed render's output directory so aborted
// runs never leave orphaned frame folders that look playable but are not.
func cleanupPartialAnim(dir string) {
	if err := os.RemoveAll(dir); err != nil {
		fmt.Printf("[Animation] WARNING: could not remove partial output %s: %v\n", dir, err)
	}
}

// ============================================================================
// SHAPE MORPH (structure-field in-betweening)
// ============================================================================
//
// Crossfade and parameter morphs both read as dissolves: two cells are
// independent realizations, so nothing in A is ever MOVED to where it
// belongs in B, and blending two finished images shows both at once.
// Shape morph works like a traditional in-betweener instead:
//
//  1. Correspondence: a dense, smooth optical flow is estimated between
//     the two keyframes' STRUCTURE fields (the normalized luminance the
//     renderer colorizes), in both directions, with coarse-to-fine
//     Horn-Schunck on blurred, downscaled, histogram-equalized copies.
//     Equalization makes matching palette-independent: dark shapes in A
//     pair with dark shapes in B whatever their colors.
//  2. In-betweens: both structure fields are warped to time t along that
//     flow and blended into ONE field, whose contrast is then restored by
//     quantile matching (a plain average of two fields goes flat and grey
//     mid-way). Shapes therefore travel, grow, merge and split as a single
//     crisp image — never a double exposure.
//  3. Color: that one field is colorized with A's look and with B's look
//     and the two are mixed, so colors change in place on shapes that are
//     already morphing.

const (
	flowWorkSide = 320  // long side of the flow working resolution
	flowMinSide  = 10   // coarsest pyramid level (short side, px)
	flowBlur     = 2.0  // pre-blur sigma at working res: track shapes, not grain
	flowAlpha    = 0.04 // smoothness weight: higher = more rigid, coherent motion
	flowWarps    = 4    // re-linearizations per pyramid level
	flowIters    = 80   // Jacobi iterations per warp
	morphQBins   = 1024 // quantile table resolution for contrast restoration
)

// flowImage is a single matching plane (values 0..1) at flow resolution.
type flowImage struct {
	w, h int
	p    []float64
}

// flowField is a dense displacement field in flow-resolution pixels:
// for a field computed from A to B, A(p) ~ B(p + (u, v)(p)).
type flowField struct {
	w, h int
	u, v []float64
}

// parallelRows splits [0, h) into contiguous row bands across CPU cores.
func parallelRows(h int, fn func(y0, y1 int)) {
	workers := runtime.NumCPU()
	if workers > h {
		workers = h
	}
	if workers < 1 {
		workers = 1
	}
	chunk := (h + workers - 1) / workers
	var wg sync.WaitGroup
	for y0 := 0; y0 < h; y0 += chunk {
		y1 := minInt(y0+chunk, h)
		wg.Add(1)
		go func(a, b int) {
			defer wg.Done()
			fn(a, b)
		}(y0, y1)
	}
	wg.Wait()
}

// sampleBilinear reads plane p (w x h) at a fractional position, clamping
// to the border.
func sampleBilinear(p []float64, w, h int, x, y float64) float64 {
	x = clampF(x, 0, float64(w-1))
	y = clampF(y, 0, float64(h-1))
	x0, y0 := int(x), int(y)
	x1, y1 := minInt(x0+1, w-1), minInt(y0+1, h-1)
	tx, ty := x-float64(x0), y-float64(y0)
	top := p[y0*w+x0]*(1-tx) + p[y0*w+x1]*tx
	bot := p[y1*w+x0]*(1-tx) + p[y1*w+x1]*tx
	return top*(1-ty) + bot*ty
}

// bilinearAt is sampleBilinear's index and weight computation, done once
// for several same-sized fields sampled at one point; at(p) then returns
// exactly sampleBilinear(p, w, h, x, y).
type bilinearAt struct {
	i00, i01, i10, i11 int
	tx, ty             float64
}

func newBilinearAt(w, h int, x, y float64) bilinearAt {
	x = clampF(x, 0, float64(w-1))
	y = clampF(y, 0, float64(h-1))
	x0, y0 := int(x), int(y)
	x1, y1 := minInt(x0+1, w-1), minInt(y0+1, h-1)
	return bilinearAt{y0*w + x0, y0*w + x1, y1*w + x0, y1*w + x1, x - float64(x0), y - float64(y0)}
}

func (b bilinearAt) at(p []float64) float64 {
	top := p[b.i00]*(1-b.tx) + p[b.i01]*b.tx
	bot := p[b.i10]*(1-b.tx) + p[b.i11]*b.tx
	return top*(1-b.ty) + bot*b.ty
}

// mirrorCoord reflects a coordinate into [0, n-1]. Warps that reach past
// the frame edge then pick up a mirrored continuation of the texture
// instead of smearing the border pixels into streaks.
func mirrorCoord(x float64, n int) float64 {
	m := float64(n - 1)
	if m <= 0 {
		return 0
	}
	x = math.Mod(math.Abs(x), 2*m)
	if x > m {
		x = 2*m - x
	}
	return x
}

// newFlowImage box-downsamples a compact byte field (W x H) to w x h and
// histogram-equalizes it (rank / n) so matching ignores tonal differences.
func newFlowImage(lum []byte, W, H, w, h int) flowImage {
	fi := flowImage{w: w, h: h, p: make([]float64, w*h)}
	parallelRows(h, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			sy0, sy1 := y*H/h, maxInt((y+1)*H/h, y*H/h+1)
			for x := 0; x < w; x++ {
				sx0, sx1 := x*W/w, maxInt((x+1)*W/w, x*W/w+1)
				acc := 0
				for sy := sy0; sy < sy1; sy++ {
					for sx := sx0; sx < sx1; sx++ {
						acc += int(lum[sy*W+sx])
					}
				}
				fi.p[y*w+x] = float64(acc) / float64((sy1-sy0)*(sx1-sx0))
			}
		}
	})
	idx := make([]int, len(fi.p))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return fi.p[idx[a]] < fi.p[idx[b]] })
	eq := make([]float64, len(fi.p))
	for r, i := range idx {
		eq[i] = float64(r) / float64(maxInt(len(idx)-1, 1))
	}
	fi.p = eq
	return fi
}

// blurPlane is a separable Gaussian blur with clamped borders.
func blurPlane(p []float64, w, h int, sigma float64) []float64 {
	r := int(math.Ceil(sigma * 3))
	k := make([]float64, 2*r+1)
	sum := 0.0
	for i := -r; i <= r; i++ {
		k[i+r] = math.Exp(-float64(i*i) / (2 * sigma * sigma))
		sum += k[i+r]
	}
	for i := range k {
		k[i] /= sum
	}
	tmp := make([]float64, w*h)
	out := make([]float64, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			acc := 0.0
			for i := -r; i <= r; i++ {
				acc += k[i+r] * p[y*w+minInt(maxInt(x+i, 0), w-1)]
			}
			tmp[y*w+x] = acc
		}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			acc := 0.0
			for i := -r; i <= r; i++ {
				acc += k[i+r] * tmp[minInt(maxInt(y+i, 0), h-1)*w+x]
			}
			out[y*w+x] = acc
		}
	}
	return out
}

// halveFlowImage builds the next (coarser) pyramid level: light blur, then
// 2x decimation.
func halveFlowImage(fi flowImage) flowImage {
	w2, h2 := (fi.w+1)/2, (fi.h+1)/2
	out := flowImage{w: w2, h: h2, p: make([]float64, w2*h2)}
	b := blurPlane(fi.p, fi.w, fi.h, 1.0)
	for y := 0; y < h2; y++ {
		for x := 0; x < w2; x++ {
			out.p[y*w2+x] = b[minInt(2*y, fi.h-1)*fi.w+minInt(2*x, fi.w-1)]
		}
	}
	return out
}

// median3x3 suppresses isolated flow outliers between warps (the standard
// trick that keeps Horn-Schunck from smearing bad matches outward).
func median3x3(p []float64, w, h int) []float64 {
	out := make([]float64, w*h)
	var win [9]float64
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			n := 0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					win[n] = p[minInt(maxInt(y+dy, 0), h-1)*w+minInt(maxInt(x+dx, 0), w-1)]
					n++
				}
			}
			s := win[:]
			sort.Float64s(s)
			out[y*w+x] = s[4]
		}
	}
	return out
}

// computeFlow estimates the dense flow from a to b (a(p) ~ b(p + f(p))) with
// coarse-to-fine Horn-Schunck: at each pyramid level b is warped by the
// current estimate, the brightness-constancy term is re-linearized, and a
// 2x2 per-pixel system is relaxed with Jacobi sweeps.
func computeFlow(a, b flowImage) flowField {
	pa, pb := []flowImage{a}, []flowImage{b}
	for {
		last := pa[len(pa)-1]
		if minInt(last.w, last.h)/2 < flowMinSide {
			break
		}
		pa = append(pa, halveFlowImage(last))
		pb = append(pb, halveFlowImage(pb[len(pb)-1]))
	}

	a2 := flowAlpha * flowAlpha
	var u, v []float64
	pw, ph := 0, 0
	for lvl := len(pa) - 1; lvl >= 0; lvl-- {
		la, lb := pa[lvl], pb[lvl]
		w, h := la.w, la.h
		n := w * h

		// Upsample the coarser estimate (vectors scale with resolution).
		nu, nv := make([]float64, n), make([]float64, n)
		if u != nil {
			sx, sy := float64(w)/float64(pw), float64(h)/float64(ph)
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					qx, qy := (float64(x)+0.5)/sx-0.5, (float64(y)+0.5)/sy-0.5
					b := newBilinearAt(pw, ph, qx, qy) // u and v share the grid
					nu[y*w+x] = b.at(u) * sx
					nv[y*w+x] = b.at(v) * sy
				}
			}
		}
		u, v = nu, nv

		sxx, sxy, syy := make([]float64, n), make([]float64, n), make([]float64, n)
		bx, by := make([]float64, n), make([]float64, n)
		bw := make([]float64, n)
		ac, bc := la.p, lb.p
		for warp := 0; warp < flowWarps; warp++ {
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					i := y*w + x
					bw[i] = sampleBilinear(bc, w, h, float64(x)+u[i], float64(y)+v[i])
				}
			}
			for y := 0; y < h; y++ {
				ym, yp := maxInt(y-1, 0), minInt(y+1, h-1)
				for x := 0; x < w; x++ {
					xm, xp := maxInt(x-1, 0), minInt(x+1, w-1)
					i := y*w + x
					// Averaged gradients of A and warped B: symmetric and
					// more stable under large residual motion.
					ix := 0.25 * (bw[y*w+xp] - bw[y*w+xm] + ac[y*w+xp] - ac[y*w+xm])
					iy := 0.25 * (bw[yp*w+x] - bw[ym*w+x] + ac[yp*w+x] - ac[ym*w+x])
					cst := bw[i] - ac[i] - ix*u[i] - iy*v[i]
					sxx[i] = ix * ix
					sxy[i] = ix * iy
					syy[i] = iy * iy
					bx[i] = ix * cst
					by[i] = iy * cst
				}
			}

			// Jacobi relaxation of
			//   (a2+Sxx) u + Sxy v = a2*ubar - Bx
			//   Sxy u + (a2+Syy) v = a2*vbar - By
			un, vn := make([]float64, n), make([]float64, n)
			for it := 0; it < flowIters; it++ {
				parallelRows(h, func(ya, yb int) {
					for y := ya; y < yb; y++ {
						ym, yp := maxInt(y-1, 0), minInt(y+1, h-1)
						for x := 0; x < w; x++ {
							xm, xp := maxInt(x-1, 0), minInt(x+1, w-1)
							i := y*w + x
							ub := 0.25 * (u[y*w+xm] + u[y*w+xp] + u[ym*w+x] + u[yp*w+x])
							vb := 0.25 * (v[y*w+xm] + v[y*w+xp] + v[ym*w+x] + v[yp*w+x])
							rx := a2*ub - bx[i]
							ry := a2*vb - by[i]
							m11, m22, m12 := a2+sxx[i], a2+syy[i], sxy[i]
							det := m11*m22 - m12*m12
							un[i] = (m22*rx - m12*ry) / det
							vn[i] = (m11*ry - m12*rx) / det
						}
					}
				})
				u, un = un, u
				v, vn = vn, v
			}
			u = median3x3(u, w, h)
			v = median3x3(v, w, h)
		}
		pw, ph = w, h
	}
	return flowField{w: pw, h: ph, u: u, v: v}
}

// morphKey is one keyframe prepared for shape morphing: its compact
// structure field at render resolution, its clean genome (for colorizing),
// its luminance quantile table, and its exact final render.
type morphKey struct {
	g      Genome
	w, h   int
	lum    []byte
	chroma []float64 // nil when the genome has no chroma modulation
	quant  []float64 // inverse CDF of lum, morphQBins entries
	lumF   []float32 // lum unquantized, 0..1 (render version 2+ keys)
	quantF []float64 // inverse CDF of lumF on the 0..255 scale (version 2+)
	img    *image.RGBA
}

// newMorphKey renders genome g's structure field at the supersampled
// canvas (cw x ch, matching renderExactFramed) and its final image at
// width x height.
func newMorphKey(g Genome, width, height, cw, ch int) *morphKey {
	g.MutationRate = 0
	g.MutationPower = 0
	f := spectralField(cw, ch, rand.New(rand.NewSource(g.Seed)), g)
	k := &morphKey{g: g, w: cw, h: ch, lum: make([]byte, cw*ch)}
	for y := 0; y < ch; y++ {
		copy(k.lum[y*cw:(y+1)*cw], f.lum[y*f.padW:y*f.padW+cw])
	}
	if f.chroma != nil {
		k.chroma = make([]float64, cw*ch)
		for y := 0; y < ch; y++ {
			copy(k.chroma[y*cw:(y+1)*cw], f.chroma[y*f.padW:y*f.padW+cw])
		}
	}
	var hist [256]int
	for _, v := range k.lum {
		hist[v]++
	}
	k.quant = make([]float64, morphQBins)
	cum, v := 0, 0
	n := len(k.lum)
	for q := 0; q < morphQBins; q++ {
		target := (float64(q) + 0.5) / morphQBins * float64(n)
		for v < 255 && float64(cum+hist[v]) < target {
			cum += hist[v]
			v++
		}
		k.quant[q] = float64(v)
	}
	if g.RenderVersion >= 2 {
		// Float keys: keep the unquantized field and take its quantiles
		// from a fine histogram, so morphed frames do not fall back onto
		// 256 levels.
		k.lumF = make([]float32, cw*ch)
		for y := 0; y < ch; y++ {
			copy(k.lumF[y*cw:(y+1)*cw], f.lumF[y*f.padW:y*f.padW+cw])
		}
		const fb = 1 << 16
		fh := make([]int, fb)
		for _, v := range k.lumF {
			fh[minInt(int(v*fb), fb-1)]++
		}
		k.quantF = make([]float64, morphQBins)
		cum, bin := 0, 0
		for q := 0; q < morphQBins; q++ {
			target := (float64(q) + 0.5) / morphQBins * float64(n)
			for bin < fb-1 && float64(cum+fh[bin]) < target {
				cum += fh[bin]
				bin++
			}
			k.quantF[q] = (float64(bin) + 0.5) / fb * 255
		}
	}
	k.img = colorizeField(specField{width: cw, height: ch, padW: cw, lum: k.lum, lumF: k.lumF, chroma: k.chroma},
		rand.New(rand.NewSource(g.Seed)), g)
	if cw != width || ch != height {
		k.img = downsampleImage(k.img, width, height)
	}
	return k
}

// newMorphFlows computes both flow directions between two keyframes.
func newMorphFlows(a, b *morphKey) (fab, fba flowField) {
	scale := math.Min(1.0, float64(flowWorkSide)/float64(maxInt(a.w, a.h)))
	w := maxInt(int(math.Round(float64(a.w)*scale)), 1)
	h := maxInt(int(math.Round(float64(a.h)*scale)), 1)
	prep := func(k *morphKey) flowImage {
		fi := newFlowImage(k.lum, k.w, k.h, w, h)
		fi.p = blurPlane(fi.p, w, h, flowBlur)
		return fi
	}
	fa, fb := prep(a), prep(b)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); fab = computeFlow(fa, fb) }()
	go func() { defer wg.Done(); fba = computeFlow(fb, fa) }()
	wg.Wait()
	return fab, fba
}

// morphFrame synthesizes the in-between at progress t (clamped, so
// overshoot easings hold at the keyframes). Intermediate flows follow the
// quadratic approximation used for frame interpolation (Super SloMo):
//
//	F(t->0) = -(1-t)t F(0->1) + t^2 F(1->0)
//	F(t->1) = (1-t)^2 F(0->1) - t(1-t) F(1->0)
//
// Both structure fields are backward-warped to time t, blended, and
// quantile-mapped onto the interpolated tonal distribution; the single
// resulting field is colorized as A and as B and mixed by t. t=0 and t=1
// return the exact keyframe renders.
func morphFrame(a, b *morphKey, fab, fba flowField, t float64, width, height int) *image.RGBA {
	t = clampF(t, 0.0, 1.0)
	if t <= 0 {
		return a.img
	}
	if t >= 1 {
		return b.img
	}
	W, H := a.w, a.h
	n := W * H
	sx, sy := float64(W)/float64(fab.w), float64(H)/float64(fab.h)
	c01a, c10a := -(1-t)*t, t*t
	c01b, c10b := (1-t)*(1-t), -t*(1-t)
	hasChroma := a.chroma != nil || b.chroma != nil
	// Float morph only when both keys carry float fields; a legacy key
	// keeps the whole frame on the original byte path.
	useF := a.lumF != nil && b.lumF != nil

	samp := func(p []byte, x, y float64) float64 {
		x, y = mirrorCoord(x, W), mirrorCoord(y, H)
		x0, y0 := int(x), int(y)
		x1, y1 := minInt(x0+1, W-1), minInt(y0+1, H-1)
		tx, ty := x-float64(x0), y-float64(y0)
		top := float64(p[y0*W+x0])*(1-tx) + float64(p[y0*W+x1])*tx
		bot := float64(p[y1*W+x0])*(1-tx) + float64(p[y1*W+x1])*tx
		return top*(1-ty) + bot*ty
	}
	// sampLum samples a key's structure field on the 0..255 scale.
	sampLum := func(k *morphKey, x, y float64) float64 {
		if !useF {
			return samp(k.lum, x, y)
		}
		x, y = mirrorCoord(x, W), mirrorCoord(y, H)
		x0, y0 := int(x), int(y)
		x1, y1 := minInt(x0+1, W-1), minInt(y0+1, H-1)
		tx, ty := x-float64(x0), y-float64(y0)
		p := k.lumF
		top := float64(p[y0*W+x0])*(1-tx) + float64(p[y0*W+x1])*tx
		bot := float64(p[y1*W+x0])*(1-tx) + float64(p[y1*W+x1])*tx
		return (top*(1-ty) + bot*ty) * 255
	}
	sampF := func(p []float64, x, y float64) float64 {
		if p == nil {
			return 0
		}
		return sampleBilinear(p, W, H, mirrorCoord(x, W), mirrorCoord(y, H))
	}

	// 1. Warp both structure fields to time t and blend.
	blend := make([]float64, n)
	var chroma []float64
	if hasChroma {
		chroma = make([]float64, n)
	}
	parallelRows(H, func(ya, yb int) {
		for y := ya; y < yb; y++ {
			qy := (float64(y)+0.5)/sy - 0.5
			for x := 0; x < W; x++ {
				qx := (float64(x)+0.5)/sx - 0.5
				// Both flows live on one grid (newMorphFlows), so the four
				// lookups share one index/weight computation.
				ba := newBilinearAt(fab.w, fab.h, qx, qy)
				bb := ba
				if fba.w != fab.w || fba.h != fab.h {
					bb = newBilinearAt(fba.w, fba.h, qx, qy)
				}
				u01 := ba.at(fab.u) * sx
				v01 := ba.at(fab.v) * sy
				u10 := bb.at(fba.u) * sx
				v10 := bb.at(fba.v) * sy
				ax, ay := float64(x)+c01a*u01+c10a*u10, float64(y)+c01a*v01+c10a*v10
				bx, by := float64(x)+c01b*u01+c10b*u10, float64(y)+c01b*v01+c10b*v10
				i := y*W + x
				blend[i] = sampLum(a, ax, ay)*(1-t) + sampLum(b, bx, by)*t
				if hasChroma {
					chroma[i] = sampF(a.chroma, ax, ay)*(1-t) + sampF(b.chroma, bx, by)*t
				}
			}
		}
	})

	// 2. Contrast restoration: rank the blended field (fine histogram) and
	// map each rank onto the quantile-interpolated distribution of A and B.
	const hb = 4096
	var hist [hb + 1]int
	for _, v := range blend {
		hist[int(v/255.0*hb)]++
	}
	var cdf [hb + 1]float64
	cum := 0
	for i := 0; i <= hb; i++ {
		cdf[i] = (float64(cum) + 0.5*float64(hist[i])) / float64(n)
		cum += hist[i]
	}
	var lum []byte
	var lumF []float32
	if useF {
		lumF = make([]float32, n)
	} else {
		lum = make([]byte, n)
	}
	parallelRows(H, func(ya, yb int) {
		for i := ya * W; i < yb*W; i++ {
			q := cdf[minInt(int(blend[i]/255.0*hb), hb)] * morphQBins
			qi := minInt(int(q), morphQBins-1)
			if useF {
				v := a.quantF[qi]*(1-t) + b.quantF[qi]*t
				lumF[i] = float32(clampF(v, 0, 255) / 255)
				continue
			}
			v := a.quant[qi]*(1-t) + b.quant[qi]*t
			lum[i] = byte(clampF(v+0.5, 0, 255))
		}
	})

	// 3. Colorize the one morphed field with each keyframe's look; mix.
	f := specField{width: W, height: H, padW: W, lum: lum, lumF: lumF, chroma: chroma}
	var imgA, imgB *image.RGBA
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); imgA = colorizeField(f, rand.New(rand.NewSource(a.g.Seed)), a.g) }()
	go func() { defer wg.Done(); imgB = colorizeField(f, rand.New(rand.NewSource(b.g.Seed)), b.g) }()
	wg.Wait()
	img := blendImages(imgA, imgB, t)
	if W != width || H != height {
		img = downsampleImage(img, width, height)
	}
	return img
}

func handleRenderAnimation(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		jsonError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		// SourceCells is the ordered keyframe sequence (2..maxAnimSources
		// unique cells). When absent, the legacy A/B pair is used.
		SourceCells []int  `json:"source_cells"`
		SourceACell int    `json:"source_a_cell"`
		SourceBCell int    `json:"source_b_cell"`
		Frames      int    `json:"frames"`
		FPS         int    `json:"fps"`
		Width       int    `json:"width"`
		Height      int    `json:"height"`
		Easing      string `json:"easing"`
		Mode        string `json:"mode"` // "crossfade", "params" or "morph"
		Dir         string `json:"dir"`
		// Timeline (all optional): each cell plays its own live motion for
		// HoldFrames, then morphs to the next over TransitionFrames. With
		// either set (or Motion active) Frames is ignored and the total is
		// derived; a single source cell is then allowed (a looping clip).
		HoldFrames       int        `json:"hold_frames"`
		TransitionFrames int        `json:"transition_frames"`
		Motion           MotionSpec `json:"motion"`
		// DriftDirs (optional, one per source cell) overrides
		// Motion.DriftDir per cell, e.g. a random direction for each.
		DriftDirs []int `json:"drift_dirs"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}

	sources := req.SourceCells
	if len(sources) == 0 {
		sources = []int{req.SourceACell, req.SourceBCell}
	}
	timeline := req.HoldFrames > 0 || req.TransitionFrames > 0 || req.Motion.active()
	minSources := 2
	if timeline {
		minSources = 1
	}
	if len(sources) < minSources || len(sources) > maxAnimSources {
		jsonError(w, fmt.Sprintf("Need %d-%d source cells", minSources, maxAnimSources), http.StatusBadRequest)
		return
	}
	if timeline {
		if req.HoldFrames < 0 || req.TransitionFrames < 0 {
			jsonError(w, "Frame counts cannot be negative", http.StatusBadRequest)
			return
		}
		if len(sources) == 1 && req.HoldFrames < 2 {
			jsonError(w, "A single-cell animation needs at least 2 frames per cell", http.StatusBadRequest)
			return
		}
		if len(sources) > 1 && req.HoldFrames == 0 && req.TransitionFrames < 2 {
			jsonError(w, "Set frames per cell, or at least 2 frames per transition", http.StatusBadRequest)
			return
		}
		m := req.Motion
		if !m.valid() {
			jsonError(w, motionRangeMsg, http.StatusBadRequest)
			return
		}
		if req.DriftDirs != nil {
			if len(req.DriftDirs) != len(sources) {
				jsonError(w, "drift_dirs needs one direction per source cell", http.StatusBadRequest)
				return
			}
			for _, d := range req.DriftDirs {
				if d < 0 || d > 7 {
					jsonError(w, "Drift directions must be 0-7", http.StatusBadRequest)
					return
				}
			}
		}
		n := len(sources)
		req.Frames = n*req.HoldFrames + (n-1)*req.TransitionFrames
		if req.HoldFrames == 0 {
			req.Frames = (n-1)*req.TransitionFrames + 1 // transitions share their keyframes
		}
	}
	seen := make(map[int]bool, len(sources))
	for _, c := range sources {
		if c < 0 || c >= totalCells {
			jsonError(w, "Invalid cell index", http.StatusBadRequest)
			return
		}
		// Each cell appears in the sequence once. The legacy two-cell
		// request (A == B, a static clip) is still allowed.
		if seen[c] && len(req.SourceCells) > 0 {
			jsonError(w, fmt.Sprintf("Cell %d is used more than once", c), http.StatusBadRequest)
			return
		}
		seen[c] = true
	}
	segments := len(sources) - 1
	if !timeline && (req.Frames < len(sources) || req.Frames > 10000) {
		jsonError(w, fmt.Sprintf("Frames must be %d-10000", len(sources)), http.StatusBadRequest)
		return
	}
	if timeline && (req.Frames < 2 || req.Frames > 20000) {
		jsonError(w, fmt.Sprintf("Total frames %d out of range (2-20000)", req.Frames), http.StatusBadRequest)
		return
	}
	if req.FPS < 1 || req.FPS > 60 {
		jsonError(w, "FPS must be 1-60", http.StatusBadRequest)
		return
	}
	if req.Width <= 0 || req.Height <= 0 || req.Width > 8192 || req.Height > 8192 {
		jsonError(w, "Invalid resolution (1-8192)", http.StatusBadRequest)
		return
	}

	easing := req.Easing
	if easing == "" {
		easing = "sine"
	}
	// Bug 12 fix: validate the mode instead of letting any typo silently
	// select the expensive params path.
	mode := req.Mode
	if mode == "" {
		mode = "crossfade"
	}
	if mode != "crossfade" && mode != "params" && mode != "morph" {
		jsonError(w, "Invalid mode (use crossfade, params or morph)", http.StatusBadRequest)
		return
	}

	saveDir := req.Dir
	if saveDir == "" {
		saveDir = state.genomeDir
	}
	info, err := os.Stat(saveDir)
	if err != nil || !info.IsDir() {
		jsonError(w, "Directory does not exist", http.StatusBadRequest)
		return
	}

	state.mu.RLock()
	genomes := make([]Genome, len(sources))
	for i, c := range sources {
		genomes[i] = state.cells[c].Genome
	}
	state.mu.RUnlock()

	// Bug 3 fix: nanosecond-precision directory name so a retry never
	// interleave-writes into the same folder, and MkdirAll errors are
	// caught immediately instead of surfacing as a cryptic os.Create
	// failure halfway through the render.
	// The sequence is tagged compactly (cells 3, 1, 0 -> "c3-1-0") so a
	// full 9-cell chain stays short; the metadata JSON has the details.
	timestamp := time.Now().Format("20060102_150405")
	cellTags := make([]string, len(sources))
	for i, c := range sources {
		cellTags[i] = strconv.Itoa(c)
	}
	outputDir := filepath.Join(saveDir,
		fmt.Sprintf("anim_c%s_%s_%06d", strings.Join(cellTags, "-"),
			timestamp, time.Now().UnixNano()%1000000))
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		jsonError(w, "Cannot create output directory: "+err.Error(), http.StatusInternalServerError)
		return
	}

	easeFunc := getEasingFunc(easing)

	// All rendering goes through renderExactFramed: frames come out at
	// EXACTLY req.Width x req.Height at any aspect ratio (no 4:3 narrowing,
	// no letterboxing), with the same 8192-px supersampling budget guarding
	// memory. Both modes use this call, so all frames in a sequence are
	// mutually identical in size.
	renderEndpoint := func(g Genome) *image.RGBA {
		return renderExactFramed(g, req.Width, req.Height)
	}

	processedFiles := make([]string, req.Frames)
	nameFor := func(frame int) string { return fmt.Sprintf("frame_%05d.png", frame) }

	// encodeFrame writes one frame's PNG (safe for concurrent frames:
	// distinct files and processedFiles slots).
	encodeFrame := func(img *image.RGBA, frame int) error {
		f, err := os.Create(filepath.Join(outputDir, nameFor(frame)))
		if err != nil {
			return err
		}
		if err := pngFast.Encode(f, img); err != nil { // Bug 4 fix: checked
			f.Close()
			return err
		}
		if err := f.Close(); err != nil {
			return err
		}
		processedFiles[frame] = nameFor(frame)
		return nil
	}

	// writeFrame encodes one frame; on failure it removes the half-written
	// folder, reports the error and returns false.
	writeFrame := func(img *image.RGBA, frame int) bool {
		err := encodeFrame(img, frame)
		if err != nil {
			cleanupPartialAnim(outputDir)
			jsonError(w, "Failed to write frame: "+err.Error(), http.StatusInternalServerError)
			return false
		}
		return true
	}

	// segmentAt maps a global frame to its segment (sources[seg] ->
	// sources[seg+1]) and the local progress in [0,1] within it. Segments
	// share the timeline equally and each runs its own easing curve, so
	// every keyframe cell is reached exactly at a segment boundary. Since
	// Frames >= len(sources), consecutive frames never skip a segment.
	segmentAt := func(frame int) (int, float64) {
		pos := 0.0
		if req.Frames > 1 {
			pos = float64(frame) / float64(req.Frames-1) * float64(segments)
		}
		seg := int(pos)
		if seg >= segments {
			seg = segments - 1
		}
		return seg, pos - float64(seg)
	}

	if timeline {
		// TIMELINE: [cell 0 live: hold][0 -> 1: transition][cell 1 live:
		// hold] ... Live motion runs on one global clock, tau = frame /
		// period (period = hold, so a single cell loops seamlessly and every
		// hold completes whole motion cycles), straight through the
		// transitions, so nothing jumps at a boundary.
		//
		// Every frame is a pure function of (cell genomes, frame index), so
		// frames are rendered by a pool of workers, each rendering and
		// encoding its own frames (encoding overlaps other workers'
		// rendering). Output is byte-identical to a serial render.
		n, hold, trans := len(genomes), req.HoldFrames, req.TransitionFrames
		period := hold
		if period == 0 {
			period = req.Frames
		}
		tau := func(f int) float64 { return float64(f) / float64(period) }
		startOf := func(i int) int { // first frame belonging to cell i
			if hold > 0 {
				return i * (hold + trans)
			}
			return i * trans
		}
		// progress of transition frame k: with holds, both endpoints belong
		// to the holds; without, frame 0 is cell A itself.
		progress := func(k int) float64 {
			if hold > 0 {
				return float64(k+1) / float64(trans+1)
			}
			return float64(k) / float64(trans)
		}
		clean := func(g Genome) Genome {
			g.MutationRate, g.MutationPower = 0, 0
			return g
		}
		// motionOf is cell i's live motion: its own drift direction when
		// per-cell directions were given. A cell keeps its direction for
		// all its frames (holds and both transitions it takes part in), so
		// its drift stays continuous.
		motionOf := func(i int) MotionSpec {
			m := req.Motion
			if req.DriftDirs != nil {
				m.DriftDir = req.DriftDirs[i]
			}
			return m
		}
		workers := animWorkers(genomes, req.Width, req.Height)

		// Per-cell tile cache: without live flow or shape-shift (or for
		// cells that skip them) a cell's tile-space work is identical on
		// every frame.
		caches := make([]*tileCacheEntry, n)
		for i := range genomes {
			if !animSerialForTest && !req.Motion.churns(genomes[i]) {
				caches[i] = &tileCacheEntry{}
			}
		}

		// Frozen normalization: each cell's min-max / percentile range (or
		// rank quantile table) is captured at its start frame and reused
		// for all its frames (the capture render IS that first frame, so
		// it is reused too), so moving content does not pump the brightness.
		ranges := make([][2]float64, n)
		quants := make([][]float64, n)
		first := make([]*image.RGBA, n)
		parallelLimit(n, workers, func(i int) {
			g := animateGenome(clean(genomes[i]), motionOf(i), tau(startOf(i)))
			g.normCapture, g.normCaptureQuant, g.tileCache = &ranges[i], &quants[i], caches[i]
			first[i] = renderEndpoint(g)
		})
		live := func(i, f int) Genome {
			g := animateGenome(clean(genomes[i]), motionOf(i), tau(f))
			g.normFix, g.normLo, g.normHi, g.normQuant = true, ranges[i][0], ranges[i][1], quants[i]
			g.tileCache = caches[i]
			return g
		}
		renderLive := func(i, f int) *image.RGBA {
			if !req.Motion.active() || (hold > 0 && f == startOf(i)) {
				return first[i] // still cell, or the captured first frame
			}
			return renderEndpoint(live(i, f))
		}

		// Build the job list: one closure per frame, in timeline order.
		type frameJob struct {
			f      int
			render func() *image.RGBA
		}
		var jobs []frameJob
		f := 0
		add := func(render func() *image.RGBA) {
			jobs = append(jobs, frameJob{f, render})
			f++
		}
		for i := 0; i < n; i++ {
			for k := 0; k < hold; k++ {
				i, fr := i, f
				add(func() *image.RGBA { return renderLive(i, fr) })
			}
			if i == n-1 || trans == 0 {
				continue
			}
			a, b := i, i+1
			switch mode {
			case "morph":
				// Keys: A as it looks on the frame before the transition, B
				// as it looks on the frame after, so both holds meet the
				// morph seamlessly. The key canvas matches renderExactFramed
				// (same supersampling), so keys equal the hold frames. Keys
				// and flows are built once, by whichever worker needs them
				// first, and released when the transition's last frame is
				// done.
				tA, tB := f-1, f+trans
				if hold == 0 {
					tA = f
				}
				type morphSeg struct {
					once       sync.Once
					keyA, keyB *morphKey
					fab, fba   flowField
					left       int32
				}
				seg := &morphSeg{left: int32(trans)}
				setup := func() {
					ss := supersampleFactor(genomes[a], fieldTile, req.Width, req.Height)
					cw, ch := req.Width*ss, req.Height*ss
					seg.keyA = newMorphKey(live(a, tA), req.Width, req.Height, cw, ch)
					seg.keyB = newMorphKey(live(b, tB), req.Width, req.Height, cw, ch)
					seg.fab, seg.fba = newMorphFlows(seg.keyA, seg.keyB)
				}
				for k := 0; k < trans; k++ {
					e := easeFunc(progress(k))
					add(func() *image.RGBA {
						seg.once.Do(setup)
						img := morphFrame(seg.keyA, seg.keyB, seg.fab, seg.fba, e, req.Width, req.Height)
						if atomic.AddInt32(&seg.left, -1) == 0 {
							seg.keyA, seg.keyB = nil, nil // free ~hundreds of MB per transition
						}
						return img
					})
				}
			case "crossfade":
				for k := 0; k < trans; k++ {
					e, fr := clampF(easeFunc(progress(k)), 0, 1), f
					add(func() *image.RGBA { return blendImages(renderLive(a, fr), renderLive(b, fr), e) })
				}
			default: // params
				classic, structDiffers, fadeOnly := pairBridge(genomes[a], genomes[b])
				for k := 0; k < trans; k++ {
					e, fr := clampF(easeFunc(progress(k)), 0, 1), f
					add(func() *image.RGBA {
						if fadeOnly {
							return blendImages(renderLive(a, fr), renderLive(b, fr), e)
						}
						ga, gb := live(a, fr), live(b, fr)
						interp := interpolateGenomes(ga, gb, e)
						interp.tileCache = nil // interpolated genes: nothing to share
						interp.normLo = ga.normLo + (gb.normLo-ga.normLo)*e
						interp.normHi = ga.normHi + (gb.normHi-ga.normHi)*e
						if classic {
							interp.phaseTo, interp.phaseBlend = gb.phases(), e
						}
						img := renderEndpoint(interp)
						if structDiffers {
							if frac := (e - 0.7) / 0.3; frac > 0 {
								img = blendImages(img, renderLive(b, fr), math.Pow(frac, 1.5))
							}
						}
						return img
					})
				}
			}
		}
		if hold == 0 && f < req.Frames { // transitions only: end on the last cell
			fr := f
			add(func() *image.RGBA { return renderLive(n-1, fr) })
		}

		// Run the jobs: workers pull frames in timeline order; the first
		// write error stops the pool and is reported once.
		next := make(chan frameJob)
		var failMu sync.Mutex
		var failErr error
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for job := range next {
					failMu.Lock()
					failed := failErr != nil
					failMu.Unlock()
					if failed {
						continue
					}
					if err := encodeFrame(job.render(), job.f); err != nil {
						failMu.Lock()
						if failErr == nil {
							failErr = err
						}
						failMu.Unlock()
					}
				}
			}()
		}
		for _, job := range jobs {
			next <- job
		}
		close(next)
		wg.Wait()
		if failErr != nil {
			cleanupPartialAnim(outputDir)
			jsonError(w, "Failed to write frame: "+failErr.Error(), http.StatusInternalServerError)
			return
		}
	} else if mode == "morph" {
		// SHAPE MORPH MODE: prepare each keyframe's structure field once,
		// compute the flow pair for the current segment, then synthesize
		// every in-between (see morphFrame). Only the current segment's
		// two keyframes are held, and B is reused as the next A. Fields
		// use the same 2x supersampled canvas as renderExactFramed, so
		// keyframes match the other modes' endpoint renders exactly.
		cw, ch := req.Width*2, req.Height*2
		if cw > 8192 || ch > 8192 {
			cw, ch = req.Width, req.Height
		}

		curSeg := -1
		var keyA, keyB *morphKey
		var fab, fba flowField
		for frame := 0; frame < req.Frames; frame++ {
			seg, t := segmentAt(frame)
			if seg != curSeg {
				if keyB != nil && seg == curSeg+1 {
					keyA = keyB
				} else {
					keyA = newMorphKey(genomes[seg], req.Width, req.Height, cw, ch)
				}
				keyB = newMorphKey(genomes[seg+1], req.Width, req.Height, cw, ch)
				fab, fba = newMorphFlows(keyA, keyB)
				curSeg = seg
			}
			img := morphFrame(keyA, keyB, fab, fba, easeFunc(t), req.Width, req.Height)
			if !writeFrame(img, frame) {
				return
			}
		}
	} else if mode == "crossfade" {
		// CROSSFADE MODE: render each keyframe ONCE, blend per frame. Only
		// the current segment's two endpoints are held in memory (the end
		// of one segment is reused as the start of the next), so a long
		// chain at 8K costs no more memory than a single pair.
		clean := func(g Genome) Genome {
			g.MutationRate = 0
			g.MutationPower = 0
			return g
		}

		curSeg := -1
		var imgA, imgB *image.RGBA
		for frame := 0; frame < req.Frames; frame++ {
			seg, t := segmentAt(frame)
			if seg != curSeg {
				if imgB != nil && seg == curSeg+1 {
					imgA = imgB
				} else {
					imgA = renderEndpoint(clean(genomes[seg]))
				}
				imgB = renderEndpoint(clean(genomes[seg+1]))
				curSeg = seg
			}
			// Bug 2 fix: clamp the eased progress before blending.
			alpha := clampF(easeFunc(t), 0.0, 1.0)

			if !writeFrame(blendImages(imgA, imgB, alpha), frame) {
				return
			}
		}
	} else {
		// PARAMS MODE: interpolate genome parameters with frozen seed and
		// mutations disabled, one segment (cell pair) at a time.
		//
		// Bug 6 fix (endpoint + structural bridge): interpolateGenomes can
		// only morph the CONTINUOUS genes; Seed and the discrete genes
		// (Transform, SpikeCount, NormMode, SymmetryFold, PaletteMode,
		// Flips...) stay at the segment start's values. Bridging measures
		// guarantee each segment ENDS at its target cell and never stalls:
		//   1. the LAST frame is the true render of the final cell, and
		//   2. if the two genomes of a segment differ structurally (seed /
		//      discrete genes / different luma references), the final 30%
		//      of that segment crossfades from the interpolated frame to the
		//      true render of its target, so the discrete switch is a smooth
		//      dissolve instead of a jump, and
		//   3. the NEXT segment mirrors that dissolve over its first 30%,
		//      leaving the same true render, so chained segments meet
		//      seamlessly at the keyframe.
		type paramSegment struct {
			a, b          Genome
			classicPair   bool
			structDiffers bool
			imgTrueB      *image.RGBA
			// fadeOnly: reaction-diffusion is chaotic, so interpolating
			// any gene that feeds it (or its phases) would make the grown
			// pattern rearrange from frame to frame. Such segments cross-
			// fade between the two true renders instead.
			fadeOnly bool
			imgTrueA *image.RGBA
		}
		newParamSegment := func(a, b Genome) paramSegment {
			ps := paramSegment{a: a, b: b}
			ps.classicPair, ps.structDiffers, ps.fadeOnly = pairBridge(a, b)
			if ps.structDiffers {
				ps.imgTrueB = renderEndpoint(b)
			}
			return ps
		}

		const bridgeStart = 0.7 // final 30% of a segment dissolves into its target

		curSeg := -1
		var ps paramSegment
		// lead is the true render of the current segment's start cell when
		// the previous segment dissolved into it (measure 3), else nil.
		var lead *image.RGBA
		for frame := 0; frame < req.Frames; frame++ {
			seg, t := segmentAt(frame)
			if seg != curSeg {
				lead = nil
				if curSeg >= 0 && ps.structDiffers {
					lead = ps.imgTrueB
				}
				ps = newParamSegment(genomes[seg], genomes[seg+1])
				if ps.fadeOnly {
					ps.imgTrueA = lead
					if ps.imgTrueA == nil {
						ps.imgTrueA = renderEndpoint(ps.a)
					}
				}
				curSeg = seg
			}
			// Bug 2 fix: clamp so overshoot easings cannot extrapolate.
			eased := clampF(easeFunc(t), 0.0, 1.0)

			var img *image.RGBA
			if frame == req.Frames-1 {
				// True endpoint: the sequence always finishes at the last cell.
				if ps.imgTrueB != nil {
					img = ps.imgTrueB
				} else {
					img = renderEndpoint(ps.b)
				}
			} else if ps.fadeOnly {
				img = blendImages(ps.imgTrueA, ps.imgTrueB, eased)
			} else {
				interp := interpolateGenomes(ps.a, ps.b, eased)
				interp.MutationRate = 0
				interp.MutationPower = 0
				// Bug 8 fix: rank-equalize / percentile normalization
				// re-derives its statistics per frame and reshuffles
				// pixel ranks on tiny field changes — the dominant
				// frame-to-frame flicker source. Min-max is temporally
				// stable.
				interp.NormMode = 0
				if ps.classicPair {
					interp.phaseTo = ps.b.phases()
					interp.phaseBlend = eased
				}
				img = renderEndpoint(interp)

				if lead != nil {
					frac := 1.0 - eased/(1.0-bridgeStart)
					if frac > 0 {
						img = blendImages(img, lead, math.Pow(frac, 1.5))
					}
				}
				if ps.structDiffers {
					frac := (eased - bridgeStart) / (1.0 - bridgeStart)
					if frac > 0 {
						img = blendImages(img, ps.imgTrueB, math.Pow(frac, 1.5))
					}
				}
			}

			if !writeFrame(img, frame) {
				return
			}
		}
	}

	sequence := make([]map[string]interface{}, len(sources))
	for i, c := range sources {
		sequence[i] = map[string]interface{}{"cell": c, "genome": genomes[i]}
	}
	animJSON := map[string]interface{}{
		"timestamp":    time.Now().Format("2006-01-02 15:04:05"),
		"frames":       req.Frames,
		"fps":          req.FPS,
		"duration_sec": float64(req.Frames) / float64(req.FPS),
		"width":        req.Width,
		"height":       req.Height,
		"easing":       easing,
		"mode":         mode,
		"source_cells": sources,
		"sequence":     sequence,
		"frame_files":  processedFiles,
	}
	if timeline {
		animJSON["hold_frames"] = req.HoldFrames
		animJSON["transition_frames"] = req.TransitionFrames
		animJSON["motion"] = req.Motion
		if req.DriftDirs != nil {
			animJSON["drift_dirs"] = req.DriftDirs
		}
		if len(sources) == 1 {
			animJSON["seamless_loop"] = true
		}
	}
	if len(sources) == 2 {
		// Legacy A/B layout kept for tools reading older metadata.
		animJSON["genomes"] = map[string]interface{}{"a": sequence[0], "b": sequence[1]}
	}

	jsonPath := filepath.Join(outputDir, "animation.json")
	jsonData, _ := json.MarshalIndent(animJSON, "", "  ")
	if err := os.WriteFile(jsonPath, jsonData, 0644); err != nil {
		fmt.Printf("[Animation] WARNING: failed to write metadata: %v\n", err)
	}

	fmt.Printf("[Animation] Rendered %d frames (%s/%s) to %s\n", req.Frames, mode, easing, outputDir)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":       "rendered",
		"total_frames": req.Frames,
		"output_dir":   outputDir,
		"json_path":    jsonPath,
	})
}

// pngFast encodes every PNG the app writes (animation frames, exports,
// saved images, grid previews) at zlib's fastest level: about 3x faster
// than the default (130 vs 400 ms for a FullHD frame) for ~15% larger
// files. PNG is lossless, so the pixels are identical. An Encoder with no
// BufferPool is safe for concurrent use.
var pngFast = &png.Encoder{CompressionLevel: png.BestSpeed}

// animSerialForTest forces animations onto one worker with no tile cache,
// so tests can check the parallel, cached path is byte-identical.
var animSerialForTest bool

// animBytesPerCanvasPixel is the measured peak working set of one frame
// render per supersampled canvas pixel (~700 MB for FullHD's 3840x2160).
const animBytesPerCanvasPixel = 90

// animWorkers is how many frames an animation renders at once: one render
// keeps only ~3 of 12 cores busy, so concurrent frames gain ~2x. Bounded
// by cores (each render is itself parallel) and by memory: at most 40% of
// available RAM for working sets, so 8K frames run fewer at a time.
func animWorkers(genomes []Genome, width, height int) int {
	if animSerialForTest {
		return 1
	}
	ss := 1
	for _, g := range genomes {
		ss = maxInt(ss, supersampleFactor(g, fieldTile, width, height))
	}
	perFrame := float64(width*ss) * float64(height*ss) * animBytesPerCanvasPixel
	budget := 0.4 * float64(availableMemory())
	return clampi(int(budget/perFrame), 1, clampi(runtime.NumCPU()/3, 1, 4))
}

// availableMemory reads MemAvailable from /proc/meminfo (Linux), falling
// back to a conservative 4 GB elsewhere.
func availableMemory() int64 {
	if b, err := os.ReadFile("/proc/meminfo"); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			if f := strings.Fields(line); len(f) >= 2 && f[0] == "MemAvailable:" {
				if kb, err := strconv.ParseInt(f[1], 10, 64); err == nil {
					return kb * 1024
				}
			}
		}
	}
	return 4 << 30
}

// parallelLimit runs fn(i) for i in [0, n) with at most limit at a time.
func parallelLimit(n, limit int, fn func(i int)) {
	sem := make(chan struct{}, maxInt(limit, 1))
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int) {
			defer wg.Done()
			fn(i)
			<-sem
		}(i)
	}
	wg.Wait()
}

// pairBridge classifies a parameter-morph pair. classicPair: both are
// classic (tile) genomes, so their noise realizations can be phase-morphed.
// structDiffers: something cannot be interpolated (discrete genes, seed-
// derived extras, different references), so the morph must dissolve into
// the true render of B near its end. fadeOnly: reaction-diffusion is
// chaotic, so the whole morph is a crossfade of true renders.
func pairBridge(a, b Genome) (classicPair, structDiffers, fadeOnly bool) {
	classicPair = a.LumaRef == "" && b.LumaRef == ""
	structDiffers = a.LumaRef != b.LumaRef ||
		a.Transform != b.Transform ||
		a.SpikeCount != b.SpikeCount ||
		a.NormMode != b.NormMode ||
		a.SymmetryFold != b.SymmetryFold ||
		a.SymmetryMirror != b.SymmetryMirror ||
		a.PaletteMode != b.PaletteMode ||
		a.WarpNest != b.WarpNest ||
		a.Layer.Mode != b.Layer.Mode ||
		a.Cell.Mode != b.Cell.Mode || a.Cell.Metric != b.Cell.Metric ||
		a.Lic.Mode != b.Lic.Mode || (a.Lic.Mode > 0 && a.Lic.Width != b.Lic.Width) ||
		(a.Cell.Mode > 0 && a.Cell.cellCount() != b.Cell.cellCount()) ||
		a.AnchorCount != b.AnchorCount
	// Match-mode pairs are not phase-morphed, so any difference in their
	// phase realization must be bridged.
	if !classicPair && a.phases() != b.phases() {
		structDiffers = true
	}
	// Phase blending only morphs the noise realization. Spike positions
	// and the domain-warp flow are derived from the frame genome's Seed
	// (always the segment start's), so with different seeds they would
	// jump at the segment end unless bridged.
	if classicPair && a.Seed != b.Seed {
		usesSeedExtras := func(g Genome) bool {
			return (g.SpikeCount > 0 && g.SpikeAmp > 0.001) || g.DomainWarp > 0.001 ||
				g.Layer.Mode == 1 || // the layer mask is derived from Seed
				g.Lic.Mode == 2 // so is the swirl flow
		}
		if usesSeedExtras(a) || usesSeedExtras(b) {
			structDiffers = true
		}
	}
	if a.RD.Mode > 0 || b.RD.Mode > 0 {
		fadeOnly, structDiffers = true, true
	}
	return
}

// ============================================================================
// GLOBAL STATE
// ============================================================================

type AppState struct {
	mu        sync.RWMutex
	cells     []*Cell
	genomeDir string
}

type Cell struct {
	Locked  bool
	Image   string
	Genome  Genome
	History []Genome // undo stack, newest last, capped at 50
	// Strength is the mutation strength that bred the current genome
	// (see slotStrengths); 0 when it did not come from an evolve step.
	Strength float64
}

var state *AppState

func NewAppState(genomeDir string) *AppState {
	cs := make([]*Cell, totalCells)
	for i := range cs {
		cs[i] = &Cell{}
	}
	return &AppState{cells: cs, genomeDir: genomeDir}
}

// pushHistoryLocked records the cell's current genome on its undo stack.
// Callers must already hold state.mu.
func (s *AppState) pushHistoryLocked(idx int) {
	h := s.cells[idx].History
	if len(h) >= 50 {
		h = h[len(h)-49:]
	}
	s.cells[idx].History = append(h, s.cells[idx].Genome)
}

// ============================================================================
// HTTP HANDLERS
// ============================================================================

// jsonError replies with {"error": msg} and the given status code. The
// client reads every API response as JSON and looks for an "error" field;
// plain-text http.Error bodies made res.json() throw and hid the message.
func jsonError(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

// validCell reports whether idx addresses a grid cell, replying with a 400
// when it does not. Unchecked indices panicked the handler.
func validCell(w http.ResponseWriter, idx int) bool {
	if idx < 0 || idx >= totalCells {
		jsonError(w, "Invalid cell index", http.StatusBadRequest)
		return false
	}
	return true
}

// handleStatic serves the UI. Only the static/ directory is exposed; the
// old handler served any file under the working directory (source code,
// saved genomes and images).
func handleStatic(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/" || r.URL.Path == "/index.html" {
		http.ServeFile(w, r, "./static/index.html")
		return
	}
	if strings.HasPrefix(r.URL.Path, "/static/") {
		http.ServeFile(w, r, "."+r.URL.Path)
		return
	}
	http.NotFound(w, r)
}

func handleGrid(w http.ResponseWriter, r *http.Request) {
	state.mu.RLock()
	defer state.mu.RUnlock()
	cells := make([]map[string]interface{}, totalCells)
	for i, cell := range state.cells {
		// Ship the genome WITHOUT its embedded luma reference: match-mode
		// references are large JPEGs and would bloat every grid fetch. The
		// full genome (reference included) is served by /api/get-genome.
		pub := cell.Genome
		pub.LumaRef = ""
		cells[i] = map[string]interface{}{
			"index":    i,
			"locked":   cell.Locked,
			"image":    cell.Image,
			"genome":   pub,
			"strength": cell.Strength,
		}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"cells": cells})
}

func handleEvolve(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ClickedIndex int `json:"clicked_index"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if !validCell(w, req.ClickedIndex) {
		return
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	parentGenome := state.cells[req.ClickedIndex].Genome

	// Locked cells act as co-parents: their genes flow into the children.
	donors := make([]Genome, 0, totalCells)
	for i := 0; i < totalCells; i++ {
		if i != req.ClickedIndex && state.cells[i].Locked {
			donors = append(donors, state.cells[i].Genome)
		}
	}

	allLocked := true
	for i := 0; i < totalCells; i++ {
		if i != req.ClickedIndex && !state.cells[i].Locked {
			allLocked = false
			break
		}
	}
	if allLocked {
		jsonError(w, "All other cells are locked — unlock at least one to evolve", http.StatusConflict)
		return
	}
	free := make([]int, 0, totalCells)
	for i := 0; i < totalCells; i++ {
		if i != req.ClickedIndex && !state.cells[i].Locked {
			free = append(free, i)
		}
	}
	strengths := slotStrengths(len(free))
	var wg sync.WaitGroup
	for slot, i := range free {
		wg.Add(1)
		go func(idx int, strength float64) {
			defer wg.Done()
			mutRng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(idx)*1000000))
			childGenome := breedGenome(parentGenome, donors, mutRng, strength)
			applyFeatures(&childGenome)
			img := renderGridPreview(childGenome)
			state.pushHistoryLocked(idx)
			state.cells[idx].Genome = childGenome
			state.cells[idx].Image = imgToBase64(img)
			state.cells[idx].Strength = strength
		}(i, strengths[slot])
	}
	wg.Wait()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "evolved"})
}

func handleGenerateAll(w http.ResponseWriter, r *http.Request) {
	state.mu.Lock()
	defer state.mu.Unlock()
	var free []int
	var locked []Genome
	for i, c := range state.cells {
		if c.Locked {
			locked = append(locked, c.Genome)
		} else {
			free = append(free, i)
		}
	}
	seeded := seedPopulation(len(free), locked, rand.New(rand.NewSource(time.Now().UnixNano())))
	parallelMap(len(seeded), func(k int) {
		idx := free[k]
		state.pushHistoryLocked(idx)
		state.cells[idx].Genome = seeded[k].genome
		state.cells[idx].Image = imgToBase64(seeded[k].image)
		state.cells[idx].Strength = 0
	})
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "generated"})
}

func handleToggleLock(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Index int `json:"index"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if !validCell(w, req.Index) {
		return
	}
	state.mu.Lock()
	state.cells[req.Index].Locked = !state.cells[req.Index].Locked
	locked := state.cells[req.Index].Locked
	state.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"locked": locked})
}

func handleSaveImage(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Index int    `json:"index"`
		Size  string `json:"size"`
		Dir   string `json:"dir"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if !validCell(w, req.Index) {
		return
	}
	state.mu.RLock()
	g := state.cells[req.Index].Genome
	state.mu.RUnlock()
	parts := strings.Split(strings.ToLower(strings.TrimSpace(req.Size)), "x")
	if len(parts) != 2 {
		jsonError(w, "Invalid size format", http.StatusBadRequest)
		return
	}
	width, _ := strconv.Atoi(parts[0])
	height, _ := strconv.Atoi(parts[1])
	if width <= 0 || height <= 0 || width > 8192 || height > 8192 {
		jsonError(w, "Invalid dimensions", http.StatusBadRequest)
		return
	}
	saveDir := req.Dir
	if saveDir == "" {
		saveDir = state.genomeDir
	}
	info, err := os.Stat(saveDir)
	if err != nil || !info.IsDir() {
		jsonError(w, "Directory does not exist", http.StatusBadRequest)
		return
	}
	img := renderPreviewFramed(g, width, height)
	timestamp := time.Now().Format("20060102_150405")
	filename := filepath.Join(saveDir, fmt.Sprintf("image_%04d_%dx%d_%s.png", req.Index, img.Bounds().Dx(), img.Bounds().Dy(), timestamp))
	file, err := os.Create(filename)
	if err != nil {
		jsonError(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if err := pngFast.Encode(file, img); err != nil {
		file.Close()
		jsonError(w, "PNG encode failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	if err := file.Close(); err != nil {
		jsonError(w, "Write failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "saved", "path": filename})
}

// handleRenderImage renders a cell at the requested size and streams the PNG
// back to the browser, so the client can save it with a native save dialog.
func handleRenderImage(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		jsonError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Index int    `json:"index"`
		Size  string `json:"size"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Index < 0 || req.Index >= totalCells {
		jsonError(w, "Invalid cell index", http.StatusBadRequest)
		return
	}
	parts := strings.Split(strings.ToLower(strings.TrimSpace(req.Size)), "x")
	if len(parts) != 2 {
		jsonError(w, "Invalid size format (use WxH)", http.StatusBadRequest)
		return
	}
	width, _ := strconv.Atoi(parts[0])
	height, _ := strconv.Atoi(parts[1])
	if width <= 0 || height <= 0 || width > 8192 || height > 8192 {
		jsonError(w, "Invalid dimensions", http.StatusBadRequest)
		return
	}
	state.mu.RLock()
	g := state.cells[req.Index].Genome
	state.mu.RUnlock()

	img := renderPreviewFramed(g, width, height)

	// Encode before writing headers so a failure can still be reported
	// (once the PNG body has started, the status code is already sent).
	var buf bytes.Buffer
	if err := pngFast.Encode(&buf, img); err != nil {
		jsonError(w, "PNG encode failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	// The canvas may have been narrowed to the preview's 4:3 field of view;
	// tell the client the real size so its filename is accurate.
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("X-Image-Width", strconv.Itoa(img.Bounds().Dx()))
	w.Header().Set("X-Image-Height", strconv.Itoa(img.Bounds().Dy()))
	w.Write(buf.Bytes())
}

func handleSaveParams(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Index int    `json:"index"`
		Dir   string `json:"dir"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if !validCell(w, req.Index) {
		return
	}
	state.mu.RLock()
	g := state.cells[req.Index].Genome
	state.mu.RUnlock()
	saved := SavedGenome{
		Version:   "1.3",
		CellIndex: req.Index,
		Timestamp: time.Now().Format("2006-01-02 15:04:05"),
		Genome:    g,
	}
	jsonData, _ := json.MarshalIndent(saved, "", "  ")
	saveDir := req.Dir
	if saveDir == "" {
		saveDir = state.genomeDir
	}
	timestamp := time.Now().Format("20060102_150405")
	filename := filepath.Join(saveDir, fmt.Sprintf("genome_%04d_%s.json", req.Index, timestamp))
	if err := os.WriteFile(filename, jsonData, 0644); err != nil {
		jsonError(w, "Write failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "saved", "path": filename})
}

func handleLoadParams(w http.ResponseWriter, r *http.Request) {
	// New flow: client picked the file itself and sends its contents.
	if strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
		var req struct {
			Index int         `json:"index"`
			Saved SavedGenome `json:"saved"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			jsonError(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
		if req.Index < 0 || req.Index >= totalCells {
			jsonError(w, "Invalid cell index", http.StatusBadRequest)
			return
		}
		loadGenomeIntoCell(w, req.Index, req.Saved)
		return
	}

	// Legacy flow: client passed a server-side file path.
	var req struct {
		Index int    `json:"index"`
		Path  string `json:"path"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Index < 0 || req.Index >= totalCells {
		jsonError(w, "Invalid cell index", http.StatusBadRequest)
		return
	}
	data, err := os.ReadFile(req.Path)
	if err != nil {
		jsonError(w, err.Error(), http.StatusBadRequest)
		return
	}
	var saved SavedGenome
	if err := json.Unmarshal(data, &saved); err != nil {
		jsonError(w, "Failed to parse JSON", http.StatusBadRequest)
		return
	}
	loadGenomeIntoCell(w, req.Index, saved)
}

// Shared tail of load-params: validate version, install genome, render preview.
func loadGenomeIntoCell(w http.ResponseWriter, index int, saved SavedGenome) {
	if saved.Version != "1.1" && saved.Version != "1.2" && saved.Version != "1.3" {
		jsonError(w, "Unsupported version (expected 1.1-1.3)", http.StatusBadRequest)
		return
	}
	// Render first, then install genome + image together so the cell is
	// never observed half-updated, and record the old genome so a load can
	// be undone like every other change.
	img := imgToBase64(renderGridPreview(saved.Genome))
	state.mu.Lock()
	state.pushHistoryLocked(index)
	state.cells[index].Genome = saved.Genome
	state.cells[index].Strength = 0
	state.cells[index].Image = img
	state.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "loaded"})
}

func handleUploadImage(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		jsonError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse multipart form (max 32MB)
	err := r.ParseMultipartForm(32 << 20)
	if err != nil {
		jsonError(w, "Failed to parse upload: "+err.Error(), http.StatusBadRequest)
		return
	}

	cellIndex, err := strconv.Atoi(r.FormValue("cellIndex"))
	if err != nil || cellIndex < 0 || cellIndex >= totalCells {
		jsonError(w, "Invalid cell index", http.StatusBadRequest)
		return
	}

	method := r.FormValue("method")
	if method == "" {
		method = "brute"
	}

	iterationsStr := r.FormValue("iterations")
	iterations := 500
	if iterationsStr != "" {
		if v, err := strconv.Atoi(iterationsStr); err == nil && v > 0 {
			iterations = v
		}
	}
	// Each iteration is a full render + FFT analysis; an unbounded count
	// let one request pin the server for hours.
	const maxIterations = 10000
	if iterations > maxIterations {
		iterations = maxIterations
	}

	file, header, err := r.FormFile("image")
	if err != nil {
		jsonError(w, "No image file uploaded", http.StatusBadRequest)
		return
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		jsonError(w, "Failed to decode image: "+err.Error(), http.StatusBadRequest)
		return
	}

	fmt.Printf("[Upload] Processing '%s' (%dx%d) with method '%s' (%d iterations)\n",
		header.Filename, img.Bounds().Dx(), img.Bounds().Dy(), method, iterations)

	downsampled := downsampleImage(img, imgW, imgH)

	var genome Genome
	var score float64

	switch method {
	case "brute":
		genome, score = reverseEngineerBruteForce(downsampled, iterations)
	case "hillclimb":
		genome, score = reverseEngineerHillClimb(downsampled, iterations)
	case "estimate":
		genome, score = reverseEngineerEstimate(downsampled, iterations)
	case "precise":
		genome, score = reverseEngineerPrecise(downsampled, iterations)
	case "match":
		// Pass the original resolution so the match method can embed a
		// high-detail luma reference in the genome.
		genome, score = reverseEngineerMatch(img, iterations)
	default:
		jsonError(w, "Unknown method. Use: brute, hillclimb, estimate, precise, or match", http.StatusBadRequest)
		return
	}

	genome.RenderVersion = renderVersion
	displayImg := renderGridPreview(genome)

	state.mu.Lock()
	state.pushHistoryLocked(cellIndex)
	state.cells[cellIndex].Genome = genome
	state.cells[cellIndex].Strength = 0
	state.cells[cellIndex].Image = imgToBase64(displayImg)
	state.mu.Unlock()

	similarity := math.Max(0, 100.0*(1.0-score/(255.0*255.0)))

	fmt.Printf("[Upload] Done. Score=%.2f, Similarity=%.1f%%\n", score, similarity)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":     "processed",
		"method":     method,
		"iterations": iterations,
		"similarity": similarity,
		"genome":     genome,
	})
}

// handleUndo pops the most recent genome from a cell's undo stack and
// reinstates it.
func handleUndo(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Index int `json:"index"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Index < 0 || req.Index >= totalCells {
		jsonError(w, "Invalid cell index", http.StatusBadRequest)
		return
	}
	state.mu.Lock()
	h := state.cells[req.Index].History
	if len(h) == 0 {
		state.mu.Unlock()
		jsonError(w, "Nothing to undo for this cell", http.StatusBadRequest)
		return
	}
	g := h[len(h)-1]
	state.cells[req.Index].History = h[:len(h)-1]
	state.cells[req.Index].Genome = g
	state.cells[req.Index].Strength = 0
	// Render under the lock so the image can never lag behind (or be
	// overwritten by a concurrent evolve after) the restored genome.
	state.cells[req.Index].Image = imgToBase64(renderGridPreview(g))
	state.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "undone"})
}

// handleGetGenome serves a cell's FULL genome, luma reference included.
// The grid endpoint strips the reference to keep responses light; saving
// parameters needs it, so the client fetches it on demand.
func handleGetGenome(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Index int `json:"index"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Index < 0 || req.Index >= totalCells {
		jsonError(w, "Invalid cell index", http.StatusBadRequest)
		return
	}
	state.mu.RLock()
	g := state.cells[req.Index].Genome
	state.mu.RUnlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"genome": g})
}

// ============================================================================
// POPULATION SEEDING — QUALITY FILTER + DIVERSITY
// ============================================================================

// Scouts are cheap stand-ins for grid previews: a scoutTile-sized field
// (about 1/16 of the FFT work of the real 1024 tile) rendered at half
// preview size. Frequency-relative genes stay measured against fieldTile,
// so a scout has the same spectrum, palette and structure statistics as
// the real preview; only the random phase draw differs.
const (
	scoutTileSide = 256
	scoutW        = imgW / 2
	scoutH        = imgH / 2
)

func renderScout(g Genome) *image.RGBA {
	g.MutationRate, g.MutationPower = 0, 0
	g.scoutTile = scoutTileSide
	// Supersampled like the preview it predicts (2x: the scout tile is
	// 1/4 the size at 1/2 the canvas).
	ss := 1
	if g.RenderVersion >= 4 {
		ss = supersampleFactor(g, scoutTileSide, scoutW, scoutH)
	}
	img := generateSpectralImage(scoutW*ss, scoutH*ss, rand.New(rand.NewSource(g.Seed)), g)
	if ss > 1 {
		img = downsampleImage(img, scoutW, scoutH)
	}
	return img
}

// imageStats are the measurements population seeding judges images by.
// Everything is measured in OKLab, so contrast carried by hue (purple on
// purple of equal brightness) counts as much as lightness contrast.
type imageStats struct {
	meanL, meanA, meanB float64 // mean OKLab color
	chroma              float64 // mean OKLab chroma (saturation)
	contrast            float64 // OKLab standard deviation (lightness + color)
	detail              float64 // mean OKLab gradient magnitude per pixel
	entropy             float64 // color histogram entropy, bits (16^3 bins)
	dominant            float64 // share of pixels in the most common coarse color
	coherence           float64 // orientation coherence of the L gradient, 0..1
}

func measureImage(img *image.RGBA) imageStats {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	n := float64(w * h)
	lab := make([][3]float64, w*h)
	var st imageStats
	fine := make(map[int]int)
	var coarse [512]int
	var sum, sum2 [3]float64
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			o := img.PixOffset(x, y)
			r, g, b := img.Pix[o], img.Pix[o+1], img.Pix[o+2]
			L, A, B := srgbToOKLab(float64(r)/255, float64(g)/255, float64(b)/255)
			lab[y*w+x] = [3]float64{L, A, B}
			for c, v := range lab[y*w+x] {
				sum[c] += v
				sum2[c] += v * v
			}
			st.chroma += math.Hypot(A, B)
			fine[int(r>>4)<<8|int(g>>4)<<4|int(b>>4)]++
			coarse[int(r>>5)<<6|int(g>>5)<<3|int(b>>5)]++
		}
	}
	st.meanL, st.meanA, st.meanB = sum[0]/n, sum[1]/n, sum[2]/n
	st.chroma /= n
	var variance float64
	for c := range sum {
		m := sum[c] / n
		variance += sum2[c]/n - m*m
	}
	st.contrast = math.Sqrt(math.Max(0, variance))
	for _, c := range fine {
		p := float64(c) / n
		st.entropy -= p * math.Log2(p)
	}
	for _, c := range coarse {
		st.dominant = math.Max(st.dominant, float64(c)/n)
	}

	var jxx, jyy, jxy, grad float64
	for y := 0; y < h-1; y++ {
		for x := 0; x < w-1; x++ {
			p, px, py := lab[y*w+x], lab[y*w+x+1], lab[(y+1)*w+x]
			var gx2, gy2 float64
			for c := 0; c < 3; c++ {
				gx2 += (px[c] - p[c]) * (px[c] - p[c])
				gy2 += (py[c] - p[c]) * (py[c] - p[c])
			}
			grad += math.Sqrt(gx2 + gy2)
			gx, gy := px[0]-p[0], py[0]-p[0]
			jxx, jyy, jxy = jxx+gx*gx, jyy+gy*gy, jxy+gx*gy
		}
	}
	st.detail = grad / float64((w-1)*(h-1))
	if tr := jxx + jyy; tr > 1e-12 {
		st.coherence = math.Sqrt((jxx-jyy)*(jxx-jyy)+4*jxy*jxy) / tr
	}
	return st
}

// imageQuality scores how much there is to look at, from a scout's stats.
// Calibrated by eye on random grids: below minQuality an image is a clear
// dud (washed-out flat field, one dull color, featureless blur). Contrast
// is OKLab, so a purple-on-purple marble with little lightness contrast
// still passes on its color structure.
func imageQuality(st imageStats) float64 {
	q := st.contrast
	q *= 1 + 2*math.Min(st.chroma, 0.2)   // color carries interest too
	q *= clampF(st.entropy/4, 0.5, 1.25)  // few distinct colors
	q *= 1 - math.Max(0, st.dominant-0.4) // one color floods the frame
	q *= clampF(st.detail/0.03, 0.5, 1)   // featureless blur
	return q
}

const (
	// Recalibrated by eye for supersampled (grain-free) render version 4
	// previews, which score ~0.75x the old grainy ones: grain had been
	// propping up washed-out duds. 0.02 rejects ~10% of random genomes,
	// as 0.03 did before; pale but structured images mostly pass.
	minQuality    = 0.02  // hard reject below this
	goodQuality   = 0.055 // at or above: no diversity-score penalty
	scoutsPerCell = 6     // candidate pool size per free cell
	verifyRetries = 2     // re-picks when a real preview fails the check
)

// diversityFeatures places an image in a space where Euclidean distance
// approximates "looks different": overall tone and tint, saturation,
// contrast, scale of detail, directionality, and the genes that change an
// image's character outright (symmetry, transform, palette family).
func diversityFeatures(st imageStats, g Genome) []float64 {
	f := []float64{
		st.meanL,
		3 * st.meanA,
		3 * st.meanB,
		3 * st.chroma,
		4 * st.contrast,
		clampF(math.Log(math.Max(st.detail, 1e-4)/0.01)/math.Log(20), 0, 1),
		st.coherence,
	}
	sym := 0.0
	if g.SymmetryFold >= 2 {
		sym = 0.6
	}
	f = append(f, sym)
	for k := 0; k < 4; k++ { // transform, one-hot
		v := 0.0
		if g.Transform == k {
			v = 0.4
		}
		f = append(f, v)
	}
	turing := 0.0 // and reaction-diffusion patterns
	if g.RD.Mode > 0 {
		turing = 0.4
	}
	f = append(f, turing)
	streaks := 0.0 // and LIC strokes
	if g.Lic.Mode > 0 {
		streaks = 0.4
	}
	f = append(f, streaks)
	cellular := 0.0 // so does a cellular field
	if g.Cell.Mode > 0 {
		cellular = 0.4
	}
	f = append(f, cellular)
	layered := 0.0 // a second layer changes an image's character outright
	if g.Layer.Mode > 0 {
		layered = 0.4
	}
	f = append(f, layered)
	// Palette family, one-hot. Without it max-min picking favored the RGB
	// cosine family, whose loud palettes sit at the edges of color space.
	for k := 0; k < 3; k++ {
		v := 0.0
		if g.PaletteMode == k {
			v = 0.4
		}
		f = append(f, v)
	}
	return f
}

func featureDist(a, b []float64) float64 {
	var d float64
	for i := range a {
		d += (a[i] - b[i]) * (a[i] - b[i])
	}
	return math.Sqrt(d)
}

// scoutCandidate is one candidate genome with its scout measurements.
type scoutCandidate struct {
	genome  Genome
	quality float64
	feat    []float64
}

func scoutGenome(g Genome) scoutCandidate {
	img := renderScout(g)
	var st imageStats
	stage("measure", func() { st = measureImage(img) })
	return scoutCandidate{genome: g, quality: imageQuality(st), feat: diversityFeatures(st, g)}
}

// pickDiverse greedily picks n candidates that are as different as possible
// from each other and from the fixed (locked) feature vectors: each step
// takes the candidate whose distance to its nearest already-chosen image,
// weighted by quality, is largest. Candidates below minQuality are used
// only if too few others exist. Returns indices into cands, in pick order.
func pickDiverse(cands []scoutCandidate, fixed [][]float64, n int) []int {
	chosen := append([][]float64(nil), fixed...)
	used := make([]bool, len(cands))
	var out []int
	for len(out) < n {
		best, bestScore := -1, math.Inf(-1)
		for pass := 0; pass < 2 && best < 0; pass++ {
			for i, c := range cands {
				if used[i] || (pass == 0 && c.quality < minQuality) {
					continue
				}
				w := math.Sqrt(clampF(c.quality/goodQuality, 0, 1))
				score := w
				if len(chosen) > 0 {
					near := math.Inf(1)
					for _, f := range chosen {
						near = math.Min(near, featureDist(c.feat, f))
					}
					score = near * w
				}
				if score > bestScore {
					best, bestScore = i, score
				}
			}
		}
		if best < 0 {
			break // pool exhausted
		}
		used[best] = true
		chosen = append(chosen, cands[best].feat)
		out = append(out, best)
	}
	return out
}

// parallelMap runs fn(i) for i in [0, n) on all cores.
func parallelMap(n int, fn func(i int)) {
	var wg sync.WaitGroup
	next := make(chan int)
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Wait()
}

// seededCell is a genome chosen for a fresh grid cell, with its preview.
type seededCell struct {
	genome Genome
	image  *image.RGBA
}

// seedPopulation creates n fresh genomes for "Generate all": it scouts a
// pool of scoutsPerCell random candidates per cell, rejects duds, and
// picks the n most mutually different survivors, also keeping away from
// the locked genomes in fixed. Each pick's real preview is then checked
// (a scout is a different phase draw, so a rare preview can still come
// out flat); failures are swapped for the next-best candidate.
func seedPopulation(n int, fixed []Genome, rng *rand.Rand) []seededCell {
	cands := make([]scoutCandidate, n*scoutsPerCell)
	for i := range cands {
		g := randomGenome(rng)
		applyFeatures(&g)
		cands[i].genome = g
	}
	parallelMap(len(cands), func(i int) { cands[i] = scoutGenome(cands[i].genome) })

	fixedFeat := make([][]float64, len(fixed))
	parallelMap(len(fixed), func(i int) { fixedFeat[i] = scoutGenome(fixed[i]).feat })

	picks := pickDiverse(cands, fixedFeat, n)
	out := make([]seededCell, len(picks))
	pending := make([]int, len(picks)) // slots whose preview needs (re)rendering
	for i := range pending {
		pending[i] = i
	}
	for round := 0; ; round++ {
		failed := make([]bool, len(pending))
		parallelMap(len(pending), func(k int) {
			slot := pending[k]
			img := renderGridPreview(cands[picks[slot]].genome)
			out[slot] = seededCell{genome: cands[picks[slot]].genome, image: img}
			failed[k] = imageQuality(measureImage(downsampleImage(img, scoutW, scoutH))) < minQuality
		})
		var retry []int
		for k, slot := range pending {
			if failed[k] {
				retry = append(retry, slot)
			}
		}
		if len(retry) == 0 || round == verifyRetries {
			return out
		}
		// Re-pick each failed slot against everything else that stays;
		// a slot with no passing alternative keeps its render.
		var repicked []int
		for _, slot := range retry {
			keep := append([][]float64(nil), fixedFeat...)
			for s, p := range picks {
				if s != slot {
					keep = append(keep, cands[p].feat)
				}
			}
			cands[picks[slot]].quality = -1 // never pick the failure again
			rest := make([]scoutCandidate, len(cands))
			copy(rest, cands)
			for _, p := range picks {
				rest[p].quality = -1
			}
			if alt := pickDiverse(rest, keep, 1); len(alt) == 1 && rest[alt[0]].quality >= minQuality {
				picks[slot] = alt[0]
				repicked = append(repicked, slot)
			}
		}
		if len(repicked) == 0 {
			return out
		}
		pending = repicked
	}
}

// ============================================================================
// RENDERER FEATURE TOGGLES
// ============================================================================

// FeatureToggles controls which visual features freshly generated genomes
// may use. All default to on. Toggling affects only genomes created AFTER
// the change: applyFeatures zeroes disabled genes at creation time, so
// existing images and saved genomes render exactly as before.
type FeatureToggles struct {
	Transform  bool `json:"transform"`      // nonlinear transforms (wave 1)
	Relief     bool `json:"relief"`         // relief lighting (wave 1)
	Spikes     bool `json:"spikes"`         // spectral spikes (wave 2)
	Chroma     bool `json:"chroma"`         // chroma modulation (wave 2)
	Cone       bool `json:"cone"`           // directional cone (wave 3)
	DomainWarp bool `json:"domain_warp"`    // classic-mode warp (wave 4)
	Symmetry   bool `json:"symmetry"`       // radial symmetry (wave 5)
	AnchorPal  bool `json:"anchor_palette"` // anchor-point palette (wave 5)
	LchPal     bool `json:"lch_palette"`    // OKLCH cosine palette
	Layers     bool `json:"layers"`         // two-layer composition
	Cellular   bool `json:"cellular"`       // cellular (Worley) field
	Lic        bool `json:"lic"`            // line integral convolution
	RD         bool `json:"rd"`             // reaction-diffusion
}

var (
	featureMu sync.RWMutex
	features  = FeatureToggles{
		Transform:  true,
		Relief:     true,
		Spikes:     true,
		Chroma:     true,
		Cone:       true,
		DomainWarp: true,
		Symmetry:   true,
		AnchorPal:  true,
		LchPal:     true,
		Layers:     true,
		Cellular:   true,
		Lic:        true,
		RD:         true,
	}
)

func getFeatures() FeatureToggles {
	featureMu.RLock()
	defer featureMu.RUnlock()
	return features
}

func setFeatures(f FeatureToggles) {
	featureMu.Lock()
	features = f
	featureMu.Unlock()
}

// applyFeatures zeroes genes whose feature is disabled. Call it on every
// NEWLY created genome (random or bred) so unchecking affects only new
// images; the caller must not hold state.mu.
func applyFeatures(g *Genome) {
	f := getFeatures()
	if !f.Transform {
		g.Transform = 0
	}
	if !f.Relief {
		g.ReliefStrength = 0
	}
	if !f.Spikes {
		g.SpikeCount = 0
	}
	if !f.Chroma {
		g.ChromaStrength = 0
	}
	if !f.Cone {
		g.ConeWidth = 1.0
	}
	if !f.DomainWarp {
		g.DomainWarp = 0
	}
	if !f.Symmetry {
		g.SymmetryFold = 0
	}
	if !f.AnchorPal && g.PaletteMode == 1 {
		g.PaletteMode = 0
	}
	if !f.RD {
		g.RD.Mode = 0
	}
	if !f.Lic {
		g.Lic.Mode = 0
	}
	if !f.Cellular {
		g.Cell.Mode = 0
	}
	if !f.Layers {
		g.Layer.Mode = 0
	}
	if !f.LchPal && g.PaletteMode == 2 {
		g.PaletteMode = 0
	}
}

func handleFeatures(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	switch r.Method {
	case "GET":
		json.NewEncoder(w).Encode(getFeatures())
	case "POST":
		var f FeatureToggles
		if err := json.NewDecoder(r.Body).Decode(&f); err != nil {
			jsonError(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
		setFeatures(f)
		json.NewEncoder(w).Encode(getFeatures())
	default:
		jsonError(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// ============================================================================
// MAIN
// ============================================================================

func main() {
	genomeDir, _ := os.Getwd()
	os.MkdirAll(genomeDir, 0755)

	state = NewAppState(genomeDir)

	// Initialize grid with a filtered, diverse random population
	state.mu.Lock()
	for i, c := range seedPopulation(totalCells, nil, rand.New(rand.NewSource(time.Now().UnixNano()))) {
		state.cells[i].Genome = c.genome
		state.cells[i].Image = imgToBase64(c.image)
	}
	state.mu.Unlock()

	// Routes
	http.HandleFunc("/", handleStatic)
	http.HandleFunc("/api/grid", handleGrid)
	http.HandleFunc("/api/evolve", handleEvolve)
	http.HandleFunc("/api/generate-all", handleGenerateAll)
	http.HandleFunc("/api/toggle-lock", handleToggleLock)
	http.HandleFunc("/api/save-image", handleSaveImage)
	http.HandleFunc("/api/save-params", handleSaveParams)
	http.HandleFunc("/api/load-params", handleLoadParams)
	http.HandleFunc("/api/upload-image", handleUploadImage)
	http.HandleFunc("/api/render-animation", handleRenderAnimation)
	http.HandleFunc("/api/render-image", handleRenderImage)
	http.HandleFunc("/api/undo", handleUndo)
	http.HandleFunc("/api/get-genome", handleGetGenome)
	http.HandleFunc("/api/features", handleFeatures)
	http.HandleFunc("/api/preview-motion", handlePreviewMotion)
	registerSessionRoutes()

	fmt.Printf("Genetic Image Evolution Lab\n")
	fmt.Printf("Server listening on http://localhost:%d\n", Port)
	fmt.Printf("Press Ctrl+C to stop\n")

	if err := http.ListenAndServe(fmt.Sprintf(":%d", Port), nil); err != nil {
		fmt.Println("Server error:", err)
	}
}
