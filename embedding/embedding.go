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

// Package embedding provides types and functions to represent a word embedding
// model and to consume models encoded in multiple formats.
// embedding can consume models in the following formats:
//   - plaintext (plain) e.g. "the 0.418 0.24968 ..."
//   - binary e.g. "2000000 128\nthe 0b{vector}"
package embedding

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

type Model struct {
	dim     uint
	vectors map[string][]float64
}

func newModel() *Model {
	return &Model{
		dim:     uint(0),
		vectors: make(map[string][]float64, 0),
	}
}

func (m *Model) Vector(s string) []float64 {
	return m.vectors[s]
}

func (m *Model) Dimensions() uint {
	return m.dim
}

func (m *Model) VocabularySize() int {
	return len(m.vectors)
}

func (m *Model) addLineFromPlain(l string) error {
	if m.dim == 0 {
		return errors.New("Cannot add to Model with 0 dimensions")
	}
	splits := strings.Split(l, " ")
	word := splits[0]
	splits = splits[1:]
	if uint(len(splits)) != m.dim {
		return fmt.Errorf(
			"Plaintext line has %d values but Model has %d dimensions",
			len(splits), m.dim)
	}

	vector := make([]float64, m.dim)
	for i := range m.dim {
		val, err := strconv.ParseFloat(splits[i], 64)
		if err != nil {
			return errors.Join(errors.New("Invalid plaintext float"), err)
		}
		vector[i] = val
	}
	m.vectors[word] = vector
	return nil
}

func LoadFromPlain(r io.Reader) (*Model, error) {
	reader := bufio.NewReader(r)
	line, err := reader.ReadString(byte('\n'))
	if err != nil {
		return nil, errors.New("Could not read first line in plaintext")
	}

	model := newModel()
	splits := strings.Split(line, " ")
	model.dim = uint(len(splits) - 1)
	if model.dim == 0 {
		return nil, errors.New("Zero dimensions detected in plaintext")
	}
	model.addLineFromPlain(line[:len(line)-1])
	for {
		line, err := reader.ReadString(byte('\n'))
		if err != nil {
			break
		}
		err = model.addLineFromPlain(line[:len(line)-1])
		if err != nil {
			return nil, err
		}
	}

	return model, nil
}

func LoadFromPlainFile(p string) (*Model, error) {
	file, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	return LoadFromPlain(file)
}
