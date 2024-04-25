/*

Copyright (C) 2024 Jackie Deng

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.

*/

package embedding

import "testing"

const Epsilon = 1e-9

func float64ApproxEquals(f float64, g float64) bool {
	if (f - g) > Epsilon {
		return false
	}
	return true
}

func vectorApproxEquals(v Vector, u Vector) bool {
	for i, _ := range v {
		if (v[i] - u[i]) > Epsilon {
			return false
		}
	}
	return true
}

func TestVectors(t *testing.T) {
	v := Vector{3, 4}
	t.Log(v)

	w := v.Add(Vector{5, 6})
	if !vectorApproxEquals(w, Vector{8, 10}) {
		t.Error("Vector {3, 4} + {5, 6} should equal {8, 10}")
	}

	w = v.Subtract(Vector{9, 2})
	if !vectorApproxEquals(w, Vector{-6, 2}) {
		t.Error("Vector {3, 4} - {9, 2} should equal {-6, 2}")
	}

	d := v.Dot(Vector{-4, 5})
	if !float64ApproxEquals(d, float64(8)) {
		t.Error("Vector {3, 4} dot {-4, 5} should equal 8")
	}

	m := v.Magnitude()
	if !float64ApproxEquals(m, float64(5)) {
		t.Error("Vector {3, 4} magnitude should be 5")
	}

	w = v.Normalize()
	if !vectorApproxEquals(w, Vector{0.6, 0.8}) {
		t.Error("Vector {3, 4} normalized should be {0.6, 0.8}")
	}

	c := v.CosineSimilarity(Vector{-3, -6})
	if !float64ApproxEquals(c, -0.98386991) {
		t.Error("Vectors {3, 4} and {-3, -6} should have a cosine similarity of -0.98386991")
	}
}
