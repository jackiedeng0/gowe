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
)

type Vector []float64

func (v Vector) Add(u Vector) Vector {
	w := make([]float64, len(v))
	for i, _ := range v {
		w[i] = v[i] + u[i]
	}
	return w
}

func (v Vector) Subtract(u Vector) Vector {
	w := make([]float64, len(v))
	for i, _ := range v {
		w[i] = v[i] - u[i]
	}
	return w
}

func (v Vector) Dot(u Vector) float64 {
	w := float64(0)
	for i, _ := range v {
		w += v[i] * u[i]
	}
	return w
}

func (v Vector) Magnitude() float64 {
	var m float64 = 0
	for _, val := range v {
		m += val * val
	}
	return math.Sqrt(m)
}

func (v Vector) Normalize() Vector {
	w := make([]float64, len(v))
	m := v.Magnitude()
	for i, _ := range v {
		w[i] += v[i] / m
	}
	return w
}

func (v Vector) CosineSimilarity(u Vector) float64 {
	return v.Dot(u) / (v.Magnitude() * u.Magnitude())
}
