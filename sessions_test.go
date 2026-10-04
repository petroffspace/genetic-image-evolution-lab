package main

import (
	"bytes"
	"encoding/json"
	"image"
	"net/http"
	"net/http/httptest"
	"testing"
)

func sessionCall(t *testing.T, h http.HandlerFunc, method, url string, body interface{}) map[string]interface{} {
	t.Helper()
	var rd *bytes.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	} else {
		rd = bytes.NewReader(nil)
	}
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(method, url, rd))
	var out map[string]interface{}
	json.Unmarshal(rec.Body.Bytes(), &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("%s %s: HTTP %d %v", method, url, rec.Code, out)
	}
	return out
}

// Save → change everything → restore must bring back genomes, locks,
// previews, feature toggles and the animation settings; rename and delete
// must show up in the listing.
func TestSessionRoundTrip(t *testing.T) {
	state = NewAppState(t.TempDir())
	defer setFeatures(getFeatures())

	tiny := imgToBase64(image.NewRGBA(image.Rect(0, 0, 4, 3)))
	for i, c := range state.cells {
		c.Genome = Genome{Seed: int64(100 + i), Exponent: 2, RenderVersion: renderVersion}
		c.Image = tiny
		c.Locked = i == 4
		c.Strength = 0.5
	}
	f := getFeatures()
	f.RD = false
	setFeatures(f)

	anim := map[string]interface{}{"source_cells": []int{3, 1}, "fields": map[string]string{"anim-hold": "12"}}
	saved := sessionCall(t, handleSessionSave, "POST", "/api/sessions/save", map[string]interface{}{"animation": anim})
	id := saved["id"].(string)
	if saved["name"] == "" {
		t.Fatal("empty default name")
	}

	// Scramble the workspace.
	for _, c := range state.cells {
		c.Genome = Genome{Seed: 1}
		c.Image = ""
		c.Locked = false
	}
	f.RD = true
	setFeatures(f)

	sessionCall(t, handleSessionRename, "POST", "/api/sessions/rename", map[string]string{"id": id, "name": "  my   run "})
	list := sessionCall(t, handleSessions, "GET", "/api/sessions", nil)["sessions"].([]interface{})
	if len(list) != 1 || list[0].(map[string]interface{})["name"] != "my run" {
		t.Fatalf("listing after rename: %v", list)
	}

	res := sessionCall(t, handleSessionRestore, "POST", "/api/sessions/restore", map[string]string{"id": id})
	for i, c := range state.cells {
		if c.Genome.Seed != int64(100+i) || c.Locked != (i == 4) || c.Strength != 0.5 || c.Image != tiny {
			t.Fatalf("cell %d not restored: %+v", i, c)
		}
		if len(c.History) != 1 || c.History[0].Seed != 1 {
			t.Fatalf("cell %d: restore not undoable", i)
		}
	}
	if getFeatures().RD {
		t.Fatal("feature toggles not restored")
	}
	if a := res["animation"].(map[string]interface{}); a["fields"].(map[string]interface{})["anim-hold"] != "12" {
		t.Fatalf("animation not returned: %v", a)
	}

	rec := httptest.NewRecorder()
	handleSessionThumb(rec, httptest.NewRequest("GET", "/api/sessions/thumb?id="+id, nil))
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "image/jpeg" {
		t.Fatalf("thumbnail: HTTP %d %s", rec.Code, rec.Header().Get("Content-Type"))
	}

	sessionCall(t, handleSessionDelete, "POST", "/api/sessions/delete", map[string]string{"id": id})
	if list := sessionCall(t, handleSessions, "GET", "/api/sessions", nil)["sessions"].([]interface{}); len(list) != 0 {
		t.Fatalf("session not deleted: %v", list)
	}
}

func TestSessionRejectsBadIDs(t *testing.T) {
	state = NewAppState(t.TempDir())
	for _, id := range []string{"", "..", "../etc", "20260101-000000-zzzzzz", "20260101-000000-abcdef/../x"} {
		rec := httptest.NewRecorder()
		b, _ := json.Marshal(map[string]string{"id": id})
		handleSessionDelete(rec, httptest.NewRequest("POST", "/api/sessions/delete", bytes.NewReader(b)))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("delete %q: HTTP %d, want 400", id, rec.Code)
		}
	}
}
