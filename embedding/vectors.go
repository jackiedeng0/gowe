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

import (
	"math"
	"unsafe"
)

type FloatVector[F float32 | float64] struct {
	scalars []F
}

func (v FloatVector[F]) Add(u FloatVector[F]) FloatVector[F] {
	w := make([]F, len(v.scalars))
	for i, _ := range v.scalars {
		w[i] = v.scalars[i] + u.scalars[i]
	}
	return FloatVector[F]{
		scalars: w,
	}
}

func (v FloatVector[F]) Subtract(u FloatVector[F]) FloatVector[F] {
	w := make([]F, len(v.scalars))
	for i, _ := range v.scalars {
		w[i] = v.scalars[i] - u.scalars[i]
	}
	return FloatVector[F]{
		scalars: w,
	}
}

func (v FloatVector[F]) Dot(u FloatVector[F]) float64 {
	d := float64(0)
	for i, _ := range v.scalars {
		d += float64(v.scalars[i]) * float64(u.scalars[i])
	}
	return d
}

func (v FloatVector[F]) Magnitude() float64 {
	m := float64(0)
	for _, val := range v.scalars {
		m += float64(val) * float64(val)
	}
	return math.Sqrt(float64(m))
}

func (v FloatVector[F]) Normalize() FloatVector[F] {
	w := make([]F, len(v.scalars))
	m := F(v.Magnitude())
	for i, _ := range v.scalars {
		w[i] += v.scalars[i] / m
	}
	return FloatVector[F]{
		scalars: w,
	}
}

// Fused-loop implementation of CosineSimilarity
func (v FloatVector[F]) CosineSimilarity(u FloatVector[F]) float64 {
	d, mV, mU := float64(0), float64(0), float64(0)
	for i, _ := range v.scalars {
		d += float64(v.scalars[i]) * float64(u.scalars[i])
		mV += float64(v.scalars[i]) * float64(v.scalars[i])
		mU += float64(u.scalars[i]) * float64(u.scalars[i])
	}
	return d / math.Sqrt(mV*mU)
}

// IntVector is used for quantized representations of FloatVectors, the shift
// value represents how many bits shifted the integer is from the underlying
// float's real magnitude value, in other words, the number of bits that can
// the decimal portion of the scalars.
type IntVector[I int8 | int16 | int32] struct {
	scalars []I
	shift   uint8
}

// Never operate on IntVectors of different shifts, this operation is designed
// to be fast so it doesn't check it.
func (v IntVector[I]) Add(u IntVector[I]) IntVector[I] {
	w := make([]I, len(v.scalars))
	for i, _ := range v.scalars {
		w[i] = v.scalars[i] + u.scalars[i]
	}
	return IntVector[I]{
		scalars: w,
		shift:   v.shift,
	}
}

func (v IntVector[I]) Subtract(u IntVector[I]) IntVector[I] {
	w := make([]I, len(v.scalars))
	for i, _ := range v.scalars {
		w[i] = v.scalars[i] - u.scalars[i]
	}
	return IntVector[I]{
		scalars: w,
		shift:   v.shift,
	}
}

func (v IntVector[I]) Dot(u IntVector[I]) float64 {
	w := int64(0)
	for i, _ := range v.scalars {
		w += int64(v.scalars[i]) * int64(u.scalars[i])
	}
	scale := float64(int64(1) << (v.shift + u.shift))
	return float64(w) / scale
}

func (v IntVector[I]) Magnitude() float64 {
	m := int64(0)
	for _, val := range v.scalars {
		m += int64(val) * int64(val)
	}
	scale := float64(int64(1) << (v.shift * 2))
	return math.Sqrt(float64(m) / scale)
}

func (v IntVector[I]) Normalize() IntVector[I] {
	f := DequantizeIntVector[float64](v)
	nF := f.Normalize()
	return QuantizeFloatVector[I](nF, 1)
}

// Fused-loop implementation of CosineSimilarity
func (v IntVector[int32]) CosineSimilarity(u IntVector[int32]) float64 {
	d, mV, mU := int64(0), int64(0), int64(0)
	for i, _ := range v.scalars {
		d += int64(v.scalars[i]) * int64(u.scalars[i])
		mV += int64(v.scalars[i]) * int64(v.scalars[i])
		mU += int64(u.scalars[i]) * int64(u.scalars[i])
	}
	// The shifts actually balance out in this equation so we don't need to
	// rescale the result
	return float64(d) / math.Sqrt(float64(mV)*float64(mU))
}

// FloatsToIntVector quantizes with respect to an expected maximum magnitude
// (must be positive) and then give it a bit more room for vector operations
//
// For example, if we want to convert a float64 vector [-2.89, 0.2] to an int8
// vector, we might want to use a maximum magnitude of 3.00 so we call
//
//	v := FloatVector[float64]{scalars: []float64{-2.89, 0.2},}
//	qV := QuantizeFloatVector[int8](v, 3.0)
//
// So we take the ceiling of Log2(3.0) which is ceil(1.58) = 2, so we need at
// 2 bits to represent the whole number component of the magnitude.
// We then reserve 2 extra bits for math and 1 bit for the sign, and our
// resulting quantized ints will look like this:
//
//	| sign[1] | extra[2] | whole[2] | fractional[3]
//
// Our resulting shift will be 3 with a decimal precision of 2^-3 = 0.125
// If we convert the int8 value back to float64, we should get:
//
//	[-2.875, 0.25]
func QuantizeFloatVector[I int8 | int16 | int32, F float32 | float64](
	v FloatVector[F], maxMagnitude float64) IntVector[I] {

	var precision I
	shift := uint8(unsafe.Sizeof(precision)*8) -
		uint8(math.Ceil(math.Log2(maxMagnitude))) - uint8(3)
	scale := F(int64(1) << shift)
	qScalars := make([]I, len(v.scalars))
	for i, _ := range v.scalars {
		qScalars[i] = I(v.scalars[i] * scale)
	}
	return IntVector[I]{
		scalars: qScalars,
		shift:   shift,
	}
}

func DequantizeIntVector[F float32 | float64, I int8 | int16 | int32](
	v IntVector[I]) FloatVector[F] {

	dScalars := make([]F, len(v.scalars))
	for i, _ := range v.scalars {
		dScalars[i] = F(v.scalars[i] >> v.shift)
	}
	return FloatVector[F]{
		scalars: dScalars,
	}
}
