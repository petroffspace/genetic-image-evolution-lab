package main

import "testing"

// Regression test: all monotone easings must start at 0, end at 1,
// stay inside [0,1], and never decrease.
// Overshoot easings (back, elastic, bounce) are deliberately excluded —
// going outside [0,1] is their designed behavior, handled by clamping.
// Run: go test -run TestEasings -v
func TestEasings(t *testing.T) {
	names := []string{"linear", "insin", "outsin", "inoutsin",
		"inquad", "outquad", "inoutquad", "incubic", "outcubic", "inoutcubic",
		"inquart", "outquart", "inoutquart", "inquint", "outquint",
		"inoutquint", "expoin", "expoout", "circlein", "circleout", "gauss"}

	for _, name := range names {
		f := getEasingFunc(name)

		if v := f(0); v < -1e-9 || v > 1e-6 {
			t.Errorf("%s: f(0) = %v, want 0", name, v)
		}
		if v := f(1); v < 1-1e-6 || v > 1+1e-6 {
			t.Errorf("%s: f(1) = %v, want 1", name, v)
		}

		prev := 0.0
		for i := 1; i <= 100; i++ {
			tv := float64(i) / 100.0
			v := f(tv)
			if v < prev-1e-9 {
				t.Errorf("%s: non-monotone at t=%.2f (%v < %v)", name, tv, v, prev)
			}
			if v < -1e-9 || v > 1+1e-6 {
				t.Errorf("%s: f(%.2f) = %v outside [0,1]", name, tv, v)
			}
			prev = v
		}
	}
}
