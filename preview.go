package main

// ============================================================================
// LIVE MOTION PREVIEW — one loop of a cell's motion at grid-preview size
// ============================================================================

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/jpeg"
	"net/http"
	"runtime"
)

// Preview loops are short: previewFrames frames for one whole motion loop
// (the client plays them at a low frame rate).
const (
	previewFrames    = 24
	maxPreviewFrames = 96
)

// valid reports whether every motion setting is within its range.
func (m MotionSpec) valid() bool {
	return m.Flow >= 0 && m.Flow <= 5 && m.Drift >= 0 && m.Drift <= 8 &&
		m.Color >= 0 && m.Color <= 8 && m.Morph >= 0 && m.Morph <= 3
}

const motionRangeMsg = "Motion out of range (flow 0-5, drift 0-8, color 0-8, shape-shift 0-3)"

// renderMotionLoop renders frames evenly spaced frames of one loop of g
// under m, exactly as an animation hold renders them (mutation grain off,
// normalization frozen at the first frame, tile work shared when the
// motion leaves the structure unchanged), at width x height.
func renderMotionLoop(g Genome, m MotionSpec, frames, width, height int) []*image.RGBA {
	g.MutationRate, g.MutationPower = 0, 0
	var cache *tileCacheEntry
	if !m.churns(g) {
		cache = &tileCacheEntry{}
	}
	var rng [2]float64
	var quant []float64
	out := make([]*image.RGBA, frames)

	first := animateGenome(g, m, 0)
	first.normCapture, first.normCaptureQuant, first.tileCache = &rng, &quant, cache
	out[0] = renderExactFramed(first, width, height)

	parallelLimit(frames-1, runtime.NumCPU(), func(i int) {
		f := animateGenome(g, m, float64(i+1)/float64(frames))
		f.normFix, f.normLo, f.normHi, f.normQuant = true, rng[0], rng[1], quant
		f.tileCache = cache
		out[i+1] = renderExactFramed(f, width, height)
	})
	return out
}

func jpegDataURL(img image.Image) string {
	var buf bytes.Buffer
	jpeg.Encode(&buf, img, &jpeg.Options{Quality: 88})
	return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

// POST /api/preview-motion {index, motion, frames?} — one loop of the
// cell's live motion as JPEG data URLs, grid-preview sized.
func handlePreviewMotion(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Index  int        `json:"index"`
		Motion MotionSpec `json:"motion"`
		Frames int        `json:"frames"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	if !validCell(w, req.Index) {
		return
	}
	if !req.Motion.valid() {
		jsonError(w, motionRangeMsg, http.StatusBadRequest)
		return
	}
	if req.Frames == 0 {
		req.Frames = previewFrames
	}
	if req.Frames < 2 || req.Frames > maxPreviewFrames {
		jsonError(w, "Frames must be 2-96", http.StatusBadRequest)
		return
	}

	state.mu.RLock()
	g := state.cells[req.Index].Genome
	state.mu.RUnlock()

	imgs := renderMotionLoop(g, req.Motion, req.Frames, imgW, imgH)
	urls := make([]string, len(imgs))
	parallelMap(len(imgs), func(i int) { urls[i] = jpegDataURL(imgs[i]) })
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"frames": urls})
}
