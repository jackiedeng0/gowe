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
	"cmp"
	"errors"
	"slices"
)

type Model[T VectorScalar] interface {
	// Loads model from plaintext file
	FromPlainFile(p string, desc bool, opts ...interface{}) error
	// Returns vector as array of scalars for a word. Note that for IntModels,
	// this will return the shifted quantized ints.
	Vector(s string) []T
	// Returns dimensions
	Dimensions() uint
	// Returns size of vocabulary
	VocabularySize() uint
	// Returns the cosine similarity between two strings
	Similarity(s, t string) float64
}

/** Common Functions **/
type relativeWord struct {
	word       string
	similarity float64
}

func RankSimilarity[T VectorScalar, M Model[T]](m M, s string, vocab []string) []string {
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

func NNearestIn[T VectorScalar, M Model[T]](m M, s string, vocab []string, n uint) ([]string, error) {
	if n == 0 {
		return nil, errors.New("n = 0 for NNearestIn() is invalid")
	} else if n > uint(len(vocab)) {
		return nil, errors.New("n > vocabulary size for NNearestIn() is invalid")
	}

	return RankSimilarity[T](m, s, vocab)[:n], nil
}
