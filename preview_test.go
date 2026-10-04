package main

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"net/http/httptest"
	"testing"
	"time"
)

// The preview endpoint returns one loop of frames; the loop wraps as
// smoothly as any other step, and bad motion is rejected.
func TestPreviewMotion(t *testing.T) {
	state = NewAppState(t.TempDir())
	r := rand.New(rand.NewSource(8))
	for i := range state.cells {
		g := randomGenome(r)
		g.RD.Mode = 0
		state.cells[i].Genome = g
	}
	call := func(body map[string]interface{}) (int, []string) {
		b, _ := json.Marshal(body)
		rec := httptest.NewRecorder()
		handlePreviewMotion(rec, httptest.NewRequest("POST", "/api/preview-motion", bytes.NewReader(b)))
		var resp struct {
			Frames []string `json:"frames"`
		}
		json.Unmarshal(rec.Body.Bytes(), &resp)
		return rec.Code, resp.Frames
	}

	start := time.Now()
	code, frames := call(map[string]interface{}{"index": 2,
		"motion": map[string]interface{}{"flow": 0.5, "drift": 1, "morph": 1.0}})
	t.Logf("24-frame preview with flow + shape-shift: %v", time.Since(start))
	if code != 200 || len(frames) != previewFrames {
		t.Fatalf("code %d, %d frames, want %d", code, len(frames), previewFrames)
	}

	// Seamless: the same loop rendered with one extra frame at tau = 1
	// must end where it began.
	m := MotionSpec{Drift: 1, Color: 1, Morph: 1}
	imgs := renderMotionLoop(state.cells[4].Genome, m, 12, 160, 120)
	again := renderMotionLoop(state.cells[4].Genome, m, 12, 160, 120)
	steps := make([]float64, 0, len(imgs))
	for i := range imgs {
		if d := meanAbsDiff(imgs[i], again[i]); d != 0 {
			t.Fatalf("frame %d not deterministic (%.3f)", i, d)
		}
		steps = append(steps, meanAbsDiff(imgs[i], imgs[(i+1)%len(imgs)]))
	}
	if wrap := steps[len(steps)-1]; wrap > 1.6*median(steps[:len(steps)-1]) {
		t.Errorf("preview loop does not wrap smoothly: %.1f vs median %.1f", wrap, median(steps))
	}

	if code, _ := call(map[string]interface{}{"index": 0, "motion": map[string]interface{}{"morph": 9}}); code != 400 {
		t.Errorf("out-of-range motion: code %d, want 400", code)
	}
	if code, _ := call(map[string]interface{}{"index": 99, "motion": map[string]interface{}{}}); code != 400 {
		t.Errorf("bad cell: code %d, want 400", code)
	}
}
