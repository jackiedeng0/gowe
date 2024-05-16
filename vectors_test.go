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

package gowe

import "testing"

const Epsilon = 1e-9

func float64ApproxEquals(f float64, g float64) bool {
	if (f - g) > Epsilon {
		return false
	}
	return true
}

func floatVectorApprox[F float32 | float64](v FloatVector[F], u FloatVector[F]) bool {
	for i, _ := range v.scalars {
		if (v.scalars[i] - u.scalars[i]) > Epsilon {
			return false
		}
	}
	return true
}

func TestFloatVectors(t *testing.T) {
	v := FloatVector[float64]{scalars: []float64{3, 4}}

	w := v.Add(FloatVector[float64]{scalars: []float64{5, 6}})
	if !floatVectorApprox(w, FloatVector[float64]{scalars: []float64{8, 10}}) {
		t.Error("Vector {3, 4} + {5, 6} should equal {8, 10}")
	}

	w = v.Subtract(FloatVector[float64]{scalars: []float64{9, 2}})
	if !floatVectorApprox(w, FloatVector[float64]{scalars: []float64{-6, 2}}) {
		t.Error("Vector {3, 4} - {9, 2} should equal {-6, 2}")
	}

	d := v.Dot(FloatVector[float64]{scalars: []float64{-4, 5}})
	if !float64ApproxEquals(d, float64(8)) {
		t.Error("Vector {3, 4} dot {-4, 5} should equal 8")
	}

	m := v.Magnitude()
	if !float64ApproxEquals(m, float64(5)) {
		t.Error("Vector {3, 4} magnitude should be 5")
	}

	w = v.Normalize()
	if !floatVectorApprox(w, FloatVector[float64]{scalars: []float64{0.6, 0.8}}) {
		t.Error("Vector {3, 4} normalized should be {0.6, 0.8}")
	}

	c := v.CosineSimilarity(FloatVector[float64]{scalars: []float64{-3, -6}})
	if !float64ApproxEquals(c, -0.98386991) {
		t.Error("Vectors {3, 4} and {-3, -6} should have a cosine similarity of -0.98386991")
	}
}

func intVectorEquals[I int8 | int16 | int32](v IntVector[I], u IntVector[I]) bool {
	if u.shift != v.shift {
		return false
	}
	for i, _ := range v.scalars {
		if v.scalars[i] != u.scalars[i] {
			return false
		}
	}
	return true
}

func TestIntVectors(t *testing.T) {
	tShift := uint8(5)

	v := IntVector[int16]{scalars: []int16{3 << tShift, 4 << tShift}, shift: tShift}

	w := v.Add(IntVector[int16]{scalars: []int16{5 << tShift, 6 << tShift}, shift: tShift})
	if !intVectorEquals(w, IntVector[int16]{scalars: []int16{8 << tShift, 10 << tShift}, shift: tShift}) {
		t.Error("Vector {3, 4} + {5, 6} should equal {8, 10}")
	}

	w = v.Subtract(IntVector[int16]{scalars: []int16{9 << tShift, 2 << tShift}, shift: tShift})
	if !intVectorEquals(w, IntVector[int16]{scalars: []int16{-6 << tShift, 2 << tShift}, shift: tShift}) {
		t.Error("Vector {3, 4} - {9, 2} should equal {-6, 2}")
	}

	d := v.Dot(IntVector[int16]{scalars: []int16{-4 << tShift, 5 << tShift}, shift: tShift})
	if !float64ApproxEquals(d, float64(8)) {
		t.Error("Vector {3, 4} dot {-4, 5} should equal 8")
	}

	m := v.Magnitude()
	if !float64ApproxEquals(m, float64(5)) {
		t.Error("Vector {3, 4} magnitude should be 5")
	}

	w = v.Normalize()
	if !intVectorEquals(w, IntVector[int16]{scalars: []int16{4915, 6553}, shift: 13}) {
		t.Error("Vector {3, 4} normalized should be {0.6, 0.8}")
	}

	c := v.CosineSimilarity(IntVector[int16]{scalars: []int16{-3, -6}, shift: tShift})
	if !float64ApproxEquals(c, -0.98386991) {
		t.Error("Vectors {3, 4} and {-3, -6} should have a cosine similarity of -0.98386991")
	}
}

func TestQuantization(t *testing.T) {
	v1 := FloatVector[float32]{
		scalars: []float32{5, 1, -9, 12},
	}

	qt8 := QuantizeFloatVector[int8](v1, 15)
	if !intVectorEquals(qt8, IntVector[int8]{scalars: []int8{10, 2, -18, 24}, shift: 1}) {
		t.Error("FloatVector {[5, 1, -9, 12]} quantized to int8 should be {[10, 2, -15, 24] 1}")
	}
	dqt8 := DequantizeIntVector[float32](qt8)
	if !floatVectorApprox(dqt8, v1) {
		t.Error("Dequantized IntVector[int8] should equal the original FloatVector")
	}

	qt16 := QuantizeFloatVector[int16](v1, 15)
	if !intVectorEquals(qt16, IntVector[int16]{scalars: []int16{2560, 512, -4608, 6144}, shift: 9}) {
		t.Error("FloatVector {[5, 1, -9, 12]} quantized to int16 should be {[2560, 512, -4608, 6144] 9}")
	}
	dqt16 := DequantizeIntVector[float32](qt16)
	if !floatVectorApprox(dqt16, v1) {
		t.Error("Dequantized IntVector[int16] should equal the original FloatVector")
	}
}
