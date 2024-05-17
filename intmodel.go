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

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

/** IntModel **/
type IntModel[I IntScalar] struct {
	dim     uint
	vectors map[string]*IntVector[I]
}

func NewIntModel[I IntScalar]() *IntModel[I] {
	return &IntModel[I]{
		dim:     uint(0),
		vectors: make(map[string]*IntVector[I], 0),
	}
}

func (m *IntModel[I]) Vector(s string) []I {
	if _, ok := m.vectors[s]; !ok {
		return make([]I, m.dim)
	}
	return m.vectors[s].scalars
}

func (m *IntModel[I]) Dimensions() uint {
	return m.dim
}

func (m *IntModel[I]) VocabularySize() uint {
	return uint(len(m.vectors))
}

func (m *IntModel[I]) Similarity(s, t string) float64 {
	v, ok := m.vectors[s]
	if !ok {
		return 0
	}
	u, ok := m.vectors[t]
	if !ok {
		return 0
	}
	return (*v).CosineSimilarity(*u)
}

// plainLineToIntModel reads a line from reader to add a word entry, it
// returns true if successfully added and false if there is nothing left to
// read or if there was an error.
func (m *IntModel[I]) plainLineToIntModel(
	br *bufio.Reader, quantShift uint8) (bool, error) {

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

	vector := make([]float64, m.dim)
	for i := range m.dim {
		val, err := strconv.ParseFloat(splits[i], 64)
		if err != nil {
			return false, errors.Join(errors.New("Invalid plaintext float"),
				err)
		}
		vector[i] = val
	}
	qv := QuantizeFloatVector[I](FloatVector[float64]{scalars: vector},
		quantShift)
	m.vectors[word] = &qv
	return true, nil
}

func (m *IntModel[I]) FromPlainFile(
	p string, desc bool, opts ...interface{}) error {

	if len(opts) != 1 {
		return errors.New("Missing maxMagnitude (float64) as opts for " +
			"parsing plaintext into IntModel")
	}

	maxMagnitude, ok := opts[0].(float64)
	if !ok {
		return errors.New("maxMagnitude opt should be type float64")
	}

	file, err := os.Open(p)
	defer file.Close()
	if err != nil {
		return err
	}

	reader := bufio.NewReader(file)
	if desc {
		// Scan the first line if description is provided
		var size, dim uint
		n, err := fmt.Fscanln(file, &size, &dim)
		if err != nil {
			return errors.Join(
				errors.New("Could not scan description in plaintext"), err)
		}
		if n <= 2 {
			return errors.New(
				"Size and dim not found in description in plaintext")
		}
		// Save the dimension but vocabulary size will be dynamically
		// determined
		m.dim = dim
	} else {
		// Read the first line and determine dim
		line, err := reader.ReadString('\n')
		if err != nil {
			return errors.New("Could not read first line in plaintext")
		}
		splits := strings.Split(line, " ")
		m.dim = uint(len(splits) - 1)
		if m.dim == 0 {
			return errors.New("Zero dimensions detected in plaintext")
		}

		// Seek back to the beginning
		file.Seek(0, io.SeekStart)
		reader = bufio.NewReader(file)
	}

	quantShift := QuantizationShift[I](maxMagnitude)
	readMore := true
	for readMore {
		readMore, err = m.plainLineToIntModel(reader, quantShift)
		if err != nil {
			return err
		}
	}

	return nil
}
