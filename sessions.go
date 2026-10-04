package main

// ============================================================================
// SESSIONS — save / list / rename / restore / delete whole workspaces
// ============================================================================
//
// A session captures everything needed to come back to a workspace: the
// renderer feature toggles, every grid cell (genome, lock, mutation
// strength, preview image) and the Animation Studio settings (stored as
// the client sent them).
//
// On disk every session is its own directory under <genomeDir>/sessions:
//
//	sessions/<id>/session.json   metadata, features, genomes, animation
//	sessions/<id>/cell_N.png     grid previews (restore is instant)
//	sessions/<id>/thumb.jpg      3x3 contact sheet for the sidebar
//
// Previews live outside session.json so listing sessions only parses the
// small JSON files.

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

const sessionVersion = 1

// Thumbnail tile size (4:3 like the grid previews) and gap in pixels.
const (
	thumbTileW = 96
	thumbTileH = 72
	thumbGap   = 2
)

type SessionCell struct {
	Genome   Genome  `json:"genome"`
	Locked   bool    `json:"locked,omitempty"`
	Strength float64 `json:"strength,omitempty"`
}

type Session struct {
	Version int    `json:"version"`
	ID      string `json:"id"`
	Name    string `json:"name"`
	Created string `json:"created"` // RFC 3339
	// RenderVersion is the renderer that drew the stored previews; restore
	// re-renders them when the renderer has changed since.
	RenderVersion int             `json:"render_version"`
	Features      FeatureToggles  `json:"features"`
	Cells         []SessionCell   `json:"cells"`
	Animation     json.RawMessage `json:"animation,omitempty"`
}

// SessionInfo is the sidebar's view of a session.
type SessionInfo struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Created string `json:"created"`
	Locked  int    `json:"locked"`
	Thumb   string `json:"thumb"` // URL of the thumbnail
}

var (
	// sessionMu serializes session directory writes (save/rename/delete).
	sessionMu   sync.Mutex
	sessionIDRe = regexp.MustCompile(`^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$`)
)

const maxSessionName = 120

func sessionsRoot() string { return filepath.Join(state.genomeDir, "sessions") }

// sessionDir resolves a session id to its directory. Ids are validated so
// a crafted id can never escape the sessions root.
func sessionDir(id string) (string, bool) {
	if !sessionIDRe.MatchString(id) {
		return "", false
	}
	return filepath.Join(sessionsRoot(), id), true
}

func newSessionID(now time.Time) string {
	var b [3]byte
	rand.Read(b[:])
	return now.Format("20060102-150405") + "-" + hex.EncodeToString(b[:])
}

func cleanSessionName(name string) string {
	name = strings.Join(strings.Fields(name), " ")
	if r := []rune(name); len(r) > maxSessionName {
		name = string(r[:maxSessionName])
	}
	return name
}

// writeFileAtomic writes via a temp file + rename so a crash never leaves
// a half-written session.json behind.
func writeFileAtomic(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func readSession(id string) (*Session, string, error) {
	dir, ok := sessionDir(id)
	if !ok {
		return nil, "", errors.New("invalid session id")
	}
	data, err := os.ReadFile(filepath.Join(dir, "session.json"))
	if err != nil {
		return nil, "", err
	}
	var s Session
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, "", err
	}
	if s.Version > sessionVersion {
		return nil, "", errors.New("session was saved by a newer version of the app")
	}
	return &s, dir, nil
}

func writeSessionJSON(dir string, s *Session) error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return writeFileAtomic(filepath.Join(dir, "session.json"), data)
}

func decodePNGDataURL(u string) ([]byte, error) {
	const prefix = "data:image/png;base64,"
	if !strings.HasPrefix(u, prefix) {
		return nil, errors.New("not a PNG data URL")
	}
	return base64.StdEncoding.DecodeString(u[len(prefix):])
}

// sessionThumbnail lays the cell previews out as the grid shows them
// (3 columns) and encodes the sheet as a JPEG.
func sessionThumbnail(previews [][]byte) ([]byte, error) {
	const cols = 3
	rows := (len(previews) + cols - 1) / cols
	sheet := image.NewRGBA(image.Rect(0, 0,
		cols*thumbTileW+(cols-1)*thumbGap, rows*thumbTileH+(rows-1)*thumbGap))
	draw.Draw(sheet, sheet.Bounds(), &image.Uniform{color.RGBA{0x16, 0x21, 0x3e, 0xff}}, image.Point{}, draw.Src)
	for i, p := range previews {
		if p == nil {
			continue
		}
		src, err := png.Decode(bytes.NewReader(p))
		if err != nil {
			continue
		}
		tile := downsampleImage(src, thumbTileW, thumbTileH)
		x, y := (i%cols)*(thumbTileW+thumbGap), (i/cols)*(thumbTileH+thumbGap)
		draw.Draw(sheet, image.Rect(x, y, x+thumbTileW, y+thumbTileH), tile, image.Point{}, draw.Src)
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, sheet, &jpeg.Options{Quality: 85}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func listSessions() ([]SessionInfo, error) {
	entries, err := os.ReadDir(sessionsRoot())
	if errors.Is(err, os.ErrNotExist) {
		return []SessionInfo{}, nil
	}
	if err != nil {
		return nil, err
	}
	out := []SessionInfo{}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		s, _, err := readSession(e.Name())
		if err != nil {
			continue // not a session dir, or unreadable: skip it
		}
		locked := 0
		for _, c := range s.Cells {
			if c.Locked {
				locked++
			}
		}
		out = append(out, SessionInfo{
			ID:      s.ID,
			Name:    s.Name,
			Created: s.Created,
			Locked:  locked,
			Thumb:   "/api/sessions/thumb?id=" + s.ID,
		})
	}
	// Newest first; ids start with the creation timestamp.
	sort.Slice(out, func(i, j int) bool { return out[i].ID > out[j].ID })
	return out, nil
}

func writeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// decodeBody decodes a JSON request body (capped at 1 MB), replying with a
// 400 on failure.
func decodeBody(w http.ResponseWriter, r *http.Request, v interface{}) bool {
	if r.Method != "POST" {
		jsonError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return false
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(v); err != nil {
		jsonError(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
		return false
	}
	return true
}

// GET /api/sessions — the saved sessions, newest first.
func handleSessions(w http.ResponseWriter, r *http.Request) {
	list, err := listSessions()
	if err != nil {
		jsonError(w, "Failed to list sessions: "+err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]interface{}{"sessions": list})
}

// POST /api/sessions/save {name?, animation?} — snapshots the workspace.
// An empty name defaults to the current date and time.
func handleSessionSave(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name      string          `json:"name"`
		Animation json.RawMessage `json:"animation"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	if len(req.Animation) > 0 && !bytes.HasPrefix(bytes.TrimSpace(req.Animation), []byte("{")) {
		jsonError(w, "animation must be a JSON object", http.StatusBadRequest)
		return
	}

	now := time.Now()
	name := cleanSessionName(req.Name)
	if name == "" {
		name = now.Format("2006-01-02 15:04:05")
	}
	s := &Session{
		Version:       sessionVersion,
		ID:            newSessionID(now),
		Name:          name,
		Created:       now.Format(time.RFC3339),
		RenderVersion: renderVersion,
		Features:      getFeatures(),
		Cells:         make([]SessionCell, totalCells),
		Animation:     req.Animation,
	}
	images := make([]string, totalCells)
	state.mu.RLock()
	for i, c := range state.cells {
		s.Cells[i] = SessionCell{Genome: c.Genome, Locked: c.Locked, Strength: c.Strength}
		images[i] = c.Image
	}
	state.mu.RUnlock()

	previews := make([][]byte, totalCells)
	for i, u := range images {
		if p, err := decodePNGDataURL(u); err == nil {
			previews[i] = p
		}
	}
	thumb, err := sessionThumbnail(previews)
	if err != nil {
		jsonError(w, "Thumbnail failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	sessionMu.Lock()
	defer sessionMu.Unlock()
	dir, _ := sessionDir(s.ID)
	fail := func(err error) {
		os.RemoveAll(dir)
		jsonError(w, "Save failed: "+err.Error(), http.StatusInternalServerError)
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		fail(err)
		return
	}
	for i, p := range previews {
		if p == nil {
			continue
		}
		if err := os.WriteFile(filepath.Join(dir, cellPreviewName(i)), p, 0644); err != nil {
			fail(err)
			return
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "thumb.jpg"), thumb, 0644); err != nil {
		fail(err)
		return
	}
	// session.json last: a directory without it is never listed.
	if err := writeSessionJSON(dir, s); err != nil {
		fail(err)
		return
	}
	writeJSON(w, map[string]interface{}{"status": "saved", "id": s.ID, "name": s.Name})
}

func cellPreviewName(i int) string { return fmt.Sprintf("cell_%d.png", i) }

// POST /api/sessions/rename {id, name}
func handleSessionRename(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	name := cleanSessionName(req.Name)
	if name == "" {
		jsonError(w, "Name cannot be empty", http.StatusBadRequest)
		return
	}
	sessionMu.Lock()
	defer sessionMu.Unlock()
	s, dir, err := readSession(req.ID)
	if err != nil {
		jsonError(w, "Session not found: "+err.Error(), http.StatusNotFound)
		return
	}
	s.Name = name
	if err := writeSessionJSON(dir, s); err != nil {
		jsonError(w, "Rename failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]string{"status": "renamed", "name": name})
}

// POST /api/sessions/delete {id}
func handleSessionDelete(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID string `json:"id"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	dir, ok := sessionDir(req.ID)
	if !ok {
		jsonError(w, "Invalid session id", http.StatusBadRequest)
		return
	}
	sessionMu.Lock()
	defer sessionMu.Unlock()
	if err := os.RemoveAll(dir); err != nil {
		jsonError(w, "Delete failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]string{"status": "deleted"})
}

// POST /api/sessions/restore {id} — reinstates the session's feature
// toggles and grid, and returns its animation settings for the client to
// apply. Each cell's previous genome goes on its undo stack, so a restore
// can be undone cell by cell.
func handleSessionRestore(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID string `json:"id"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	s, dir, err := readSession(req.ID)
	if err != nil {
		jsonError(w, "Session not found: "+err.Error(), http.StatusNotFound)
		return
	}
	if len(s.Cells) != totalCells {
		jsonError(w, "Session has the wrong number of cells", http.StatusBadRequest)
		return
	}

	// Reuse the stored previews when the renderer is unchanged; otherwise
	// (or if one is missing) render afresh so image and genome agree.
	images := make([]string, totalCells)
	parallelMap(totalCells, func(i int) {
		if s.RenderVersion == renderVersion {
			if p, err := os.ReadFile(filepath.Join(dir, cellPreviewName(i))); err == nil {
				images[i] = "data:image/png;base64," + base64.StdEncoding.EncodeToString(p)
				return
			}
		}
		images[i] = imgToBase64(renderGridPreview(s.Cells[i].Genome))
	})

	setFeatures(s.Features)
	state.mu.Lock()
	for i, c := range s.Cells {
		state.pushHistoryLocked(i)
		state.cells[i].Genome = c.Genome
		state.cells[i].Locked = c.Locked
		state.cells[i].Strength = c.Strength
		state.cells[i].Image = images[i]
	}
	state.mu.Unlock()

	anim := s.Animation
	if len(anim) == 0 {
		anim = json.RawMessage("null")
	}
	writeJSON(w, map[string]interface{}{
		"status":    "restored",
		"name":      s.Name,
		"features":  s.Features,
		"animation": anim,
	})
}

// GET /api/sessions/thumb?id=... — the session's contact-sheet JPEG.
func handleSessionThumb(w http.ResponseWriter, r *http.Request) {
	dir, ok := sessionDir(r.URL.Query().Get("id"))
	if !ok {
		http.NotFound(w, r)
		return
	}
	// Thumbnails never change once written, so let the browser keep them.
	w.Header().Set("Cache-Control", "private, max-age=31536000, immutable")
	http.ServeFile(w, r, filepath.Join(dir, "thumb.jpg"))
}

func registerSessionRoutes() {
	http.HandleFunc("/api/sessions", handleSessions)
	http.HandleFunc("/api/sessions/save", handleSessionSave)
	http.HandleFunc("/api/sessions/rename", handleSessionRename)
	http.HandleFunc("/api/sessions/delete", handleSessionDelete)
	http.HandleFunc("/api/sessions/restore", handleSessionRestore)
	http.HandleFunc("/api/sessions/thumb", handleSessionThumb)
}
