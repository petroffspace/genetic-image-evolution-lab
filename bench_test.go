package main

import (
	"math/rand"
	"testing"
)

// Pipeline benchmarks for the two user-facing paths. For a per-stage CPU
// breakdown (spectralField and friends label their work with stage()):
//
//	go test -run XXX -bench Evolve -benchtime 16x -cpuprofile cpu.out -o bench.test
//	go tool pprof -tags bench.test cpu.out
//
// and to look inside one stage: go tool pprof -top -tagfocus=stage=synth ...

// BenchmarkGenerateAll is one "Generate all": scouting, filtering and
// picking a diverse population of 9, plus their real previews.
func BenchmarkGenerateAll(b *testing.B) {
	for i := 0; i < b.N; i++ {
		seedPopulation(9, nil, rand.New(rand.NewSource(int64(i))))
	}
}

// BenchmarkEvolve is one evolve click: 8 children bred at the grid's
// strength tiers from a random parent, previews rendered in parallel.
func BenchmarkEvolve(b *testing.B) {
	r := rand.New(rand.NewSource(7))
	parents := make([]Genome, 16)
	for i := range parents {
		parents[i] = randomGenome(r)
	}
	st := slotStrengths(8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := parents[i%len(parents)]
		kids := make([]Genome, 8)
		for j := range kids {
			kids[j] = breedGenome(p, nil, r, st[j])
		}
		parallelMap(8, func(j int) { renderGridPreview(kids[j]) })
	}
}

// BenchmarkExport renders one 1920x1080 export of a random genome.
func BenchmarkExport(b *testing.B) {
	r := rand.New(rand.NewSource(9))
	gs := make([]Genome, 8)
	for i := range gs {
		gs[i] = randomGenome(r)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		renderPreviewFramed(gs[i%len(gs)], 1920, 1080)
	}
}
