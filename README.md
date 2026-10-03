# Genetic Image Evolution Lab

Genetic Image Evolution Lab is a browser-based studio for growing abstract
artworks by **artificial evolution**. Nine images live in a grid. Click the
one you like and the others are replaced by its children: variations that
inherit its "genes", mutated and recombined. Lock favourites to breed from
several parents at once, and keep going until something beautiful appears.

Every image is computed from its genome, a small JSON recipe of a few dozen
numbers, so any result can be saved, shared, reloaded and re-rendered
**exactly**, at any size from a thumbnail to 8K. Images can also be brought to
life: each one can animate on its own as a seamless loop, or morph into the
next in a sequence, ready to encode as video.

The app is a single Go program with no dependencies: build it, run it, and
open it in your browser.

---

## Functionality

### Image generation

Each image starts as a field of noise shaped in the frequency domain, which
gives it its character: soft billows, fine filaments, brushed streaks,
crystalline lattices. On top of that, a genome can switch on:

- **Cellular patterns**: bubbles, pebbles, crack networks or stained-glass
  mosaics.
- **A second layer**: another texture shown through a mask, modulating the
  first, or laid over it.
- **Reaction-diffusion**: organic Turing patterns (spots, stripes,
  labyrinths, coral) grown out of the image itself.
- **Brush strokes**: streaks that follow the image's contours or swirl like
  hair and fur.
- **Marbling**: a deep, flowing domain warp.
- **Symmetry**: kaleidoscope and mandala folds.
- **Color**: three palette families (cosine, multi-stop ramps, and
  perceptual OKLCH palettes), plus terraces, ridges, relief lighting and
  iridescent color shifts.

Rendering is high precision throughout: smooth gradients without banding,
supersampled edges, and previews that show the same detail as full-size
exports.

### Evolution

- **Breed** by clicking an image. The clicked image is the main parent, and
  locked images join in as co-parents.
- **Gentle to wild:** the grid is filled from top to bottom with children
  that stay close to the parent (*gentle*), wander further (*normal*), or
  explore boldly (*wild*). A badge on each image shows which.
- **Coherent inheritance:** related traits (a palette, a warp, a pattern
  style) are passed on as a unit, so children stay recognisable.
- **Generate All** starts a fresh population, automatically skipping dull,
  flat or washed-out images and choosing results that differ from each other
  and from your locked images.
- **Undo** each image's changes, up to 50 steps (kept while the server runs).
- **Feature toggles** switch individual features (symmetry, cellular,
  reaction-diffusion, brush strokes, palettes, and more) on or off for
  everything generated from then on.

### Reverse engineering

Upload any picture and the app finds a genome that resembles it, either in
character (texture, palette, roughness) or in actual layout, using one of five
methods from a fast estimate to a close visual match. The result then evolves
like any other image.

### Animation Studio

- **Single-image loops:** an image animates on its own (churning texture,
  gliding drift, cycling colors), and the clip loops seamlessly.
- **Sequences** of up to 9 images: each one animates for a while, then morphs
  into the next.
- **Three transition styles:** *Shape Morph* (shapes flow and reshape into
  the next image), *Crossfade*, and *Parameter Morph* (the genome itself
  morphs).
- **Easing:** 30+ curves for the transitions.
- **Output:** any resolution from SD to 8K in five aspect ratios, written as
  a numbered PNG sequence with a metadata file, ready for FFmpeg or a video
  editor.

### Saving and sharing

- **Images**: export at any size up to 8192×8192.
- **Genomes**: save as JSON and load back later or on another machine. A
  loaded genome always reproduces exactly the same image, even after the app
  itself is updated.

## Installation

You need [Go](https://go.dev/dl/) 1.20 or newer. Nothing else is required: the
program uses only Go's standard library.

    git clone https://github.com/<you>/genetic-image-evolution-lab.git
    cd genetic-image-evolution-lab
    go build -o lab .

This builds a single executable, `lab` (`lab.exe` on Windows). To try it
without building, run `go run .` instead.

## Usage

Start the server from the project folder (it serves the web interface from
`static/` there):

    ./lab

    Genetic Image Evolution Lab
    Server listening on http://localhost:8989

Open **http://localhost:8989** in a browser. Stop the server with Ctrl+C.

### The grid

| What you want | How |
|---|---|
| Breed from an image | Click it |
| Keep an image (and use it as a co-parent) | 🔒 / 🔓 |
| Create a genome from your own picture | 📤, then pick a method |
| Load a saved genome into this cell | 📂 |
| Save this cell's genome (JSON) | 🧬 |
| Save this image at any size | 🖼️, then enter e.g. `3840x2160` |
| Undo this cell's last change | ↩️ |
| Replace all unlocked images | **Generate All** |

A typical session: press **Generate All** until something catches your eye,
click it, lock the children you like, keep clicking, and save the images and
genomes you want to keep.

### Reverse engineering methods

| Method | Best for |
|---|---|
| `brute` | Trying many random genomes and keeping the best match |
| `hillclimb` | Gradually refining one genome toward the picture |
| `estimate` | A quick analysis of texture and color, then fine-tuning (default) |
| `precise` | An instant, deterministic fit of texture and palette |
| `match` | The closest visual likeness: keeps the picture's layout |

### Making an animation

1. **Choose source cells.** One cell makes a seamless loop; **+ Add Source
   Cell** builds a sequence (✕ removes one).
2. **Set the timeline:**
   - **Frames per cell**: how long each image animates on its own.
   - **Frames per transition**: how long each morph takes.

   The total length is shown as you type.
3. **Pick the motion.** Use a preset (*Still*, *Gentle*, *Flowing*,
   *Psychedelic*), or set **Flow**, **Drift** (speed and direction) and
   **Color cycle** yourself.
4. **Pick the look:** aspect ratio, quality (SD up to 8K), transition style
   and easing.
5. **Set an output directory** (on the machine running the server; empty
   means the folder the server was started from), then click **Render PNG
   Sequence**.

**Playback FPS** doesn't change the frames; it only sets the duration shown and
the suggested encoding command. To turn the frames into a video:

    ffmpeg -framerate 24 -i anim_*/frame_%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4

Animations render several frames at once and can take a while at high
resolutions: roughly half a second per FullHD frame on a 12-core machine.

## License

MIT License

Copyright (c) 2026 petroffspace.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
