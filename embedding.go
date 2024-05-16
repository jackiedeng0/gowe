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

// Package gowe provides types and functions to represent a word embedding
// model and to consume models encoded in multiple formats.
// gowe can consume models in the following formats:
//   - plaintext (plain) e.g. "the 0.418 0.24968 ..."
//     The first line may be the vocabulary size and dim e.g. "300000 128",
//     in this case, we skip it
package gowe

import (
	"bufio"
	"cmp"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"
	"strings"
)

type Model[T float32 | float64] struct {
	dim     uint
	vectors map[string]*FloatVector[T]
}

type relativeWord struct {
	word       string
	similarity float64
}

func newModel[T float32 | float64]() *Model[T] {
	return &Model[T]{
		dim:     uint(0),
		vectors: make(map[string]*FloatVector[T], 0),
	}
}

func (m *Model[T]) Vector(s string) FloatVector[T] {
	if m.vectors[s] == nil {
		return FloatVector[T]{scalars: make([]T, m.dim)}
	}
	return *m.vectors[s]
}

func (m *Model[T]) Dimensions() uint {
	return m.dim
}

func (m *Model[T]) VocabularySize() uint {
	return uint(len(m.vectors))
}

func (m *Model[T]) Similarity(s, t string) float64 {
	return m.Vector(s).CosineSimilarity(m.Vector(t))
}

func (m *Model[T]) RankSimilarity(s string, vocab []string) []string {
	relativeWords := make([]relativeWord, len(vocab))
	for i, word := range vocab {
		relativeWords[i] = relativeWord{
			word:       word,
			similarity: m.Similarity(s, word),
		}
	}
	slices.SortFunc(relativeWords, func(a, b relativeWord) int {
		return cmp.Compare(b.similarity, a.similarity)
	})

	rankedWords := make([]string, len(vocab))
	for i, reWord := range relativeWords {
		rankedWords[i] = reWord.word
	}
	return rankedWords
}

func (m *Model[T]) NNearestIn(s string, vocab []string, n uint) ([]string, error) {
	if n == 0 {
		return nil, errors.New("n = 0 for NNearestIn() is invalid")
	} else if n > uint(len(vocab)) {
		return nil, errors.New("n > vocabulary size for NNearestIn() is invalid")
	}

	return m.RankSimilarity(s, vocab)[:n], nil
}

// addLineFromPlain reads a line from reader to add a word entry, it returns
// true if successfully added and false if there is nothing left to read or if
// there was an error.
func (m *Model[T]) addLineFromPlain(br *bufio.Reader) (bool, error) {
	word, err := br.ReadString(' ')
	word = strings.TrimRight(word, " ")
	if err != nil {
		return false, nil
	}

	line, err := br.ReadString('\n')
	splits := strings.Split(strings.TrimRight(line, "\n"), " ")
	if uint(len(splits)) != m.dim {
		return false, fmt.Errorf(
			"Plaintext line has %d values but Model has %d dimensions",
			len(splits), m.dim)
	}

	vector := make([]T, m.dim)
	// Parse differently depending on model's vector type
	switch interface{}(vector).(type) {
	case []float32:
		for i := range m.dim {
			val, err := strconv.ParseFloat(splits[i], 32)
			if err != nil {
				return false, errors.Join(errors.New("Invalid plaintext float"),
					err)
			}
			vector[i] = T(val)
		}
	case []float64:
		for i := range m.dim {
			val, err := strconv.ParseFloat(splits[i], 64)
			if err != nil {
				return false, errors.Join(errors.New("Invalid plaintext float"),
					err)
			}
			vector[i] = T(val)
		}
	default:
		return false, errors.New("Invalid type T when adding plaintext line")
	}
	m.vectors[word] = &FloatVector[T]{scalars: vector}
	return true, nil
}

func LoadFromPlain[T float32 | float64](r io.ReadSeeker, desc bool) (*Model[T], error) {
	reader := bufio.NewReader(r)
	model := newModel[T]()
	if desc {
		// Scan the first line if description is provided
		var size, dim uint
		n, err := fmt.Fscanln(r, &size, &dim)
		if err != nil {
			return nil, errors.Join(
				errors.New("Could not scan description in plaintext"), err)
		}
		if n <= 2 {
			return nil,
				errors.New("Size and dim not found in description in plaintext")
		}
		// Save the dimension but vocabulary size will be dynamically
		// determined
		model.dim = dim
	} else {
		// Read the first line and determine dim
		line, err := reader.ReadString('\n')
		if err != nil {
			return nil, errors.New("Could not read first line in plaintext")
		}
		splits := strings.Split(line, " ")
		model.dim = uint(len(splits) - 1)
		if model.dim == 0 {
			return nil, errors.New("Zero dimensions detected in plaintext")
		}

		// Seek back to the beginning
		r.Seek(0, io.SeekStart)
		reader = bufio.NewReader(r)
	}

	readMore := true
	var err error
	for readMore {
		readMore, err = model.addLineFromPlain(reader)
		if err != nil {
			return nil, err
		}
	}

	return model, nil
}

func LoadFromPlainFile[T float32 | float64](p string, desc bool) (*Model[T], error) {
	file, err := os.Open(p)
	defer file.Close()
	if err != nil {
		return nil, err
	}
	return LoadFromPlain[T](file, desc)
}
