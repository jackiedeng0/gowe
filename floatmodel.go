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
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

/** FloatModel **/
type FloatModel[F FloatScalar] struct {
	dim     uint
	vectors map[string]*FloatVector[F]
}

func NewFloatModel[F FloatScalar]() *FloatModel[F] {
	return &FloatModel[F]{
		dim:     uint(0),
		vectors: make(map[string]*FloatVector[F], 0),
	}
}

func (m *FloatModel[F]) Vector(s string) []F {
	if _, ok := m.vectors[s]; !ok {
		return make([]F, m.dim)
	}
	return m.vectors[s].scalars
}

func (m *FloatModel[F]) Dimensions() uint {
	return m.dim
}

func (m *FloatModel[F]) VocabularySize() uint {
	return uint(len(m.vectors))
}

func (m *FloatModel[F]) Similarity(s, t string) float64 {
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

// readPlainVector reads a line from reader to add a word entry, it returns
// true if successfully added and false if there is nothing left to read or if
// there was an error.
func (m *FloatModel[F]) readPlainVector(br *bufio.Reader) (bool, error) {

	word, err := br.ReadString(' ')
	if err != nil {
		return false, nil
	}
	word = strings.TrimRight(word, " ")

	line, err := br.ReadString('\n')
	splits := strings.Split(strings.TrimRight(line, "\n"), " ")
	if uint(len(splits)) != m.dim {
		return false, fmt.Errorf(
			"Plaintext line has %d values but Model has %d dimensions",
			len(splits), m.dim)
	}

	vector := make([]F, m.dim)
	// Parse differently depending on model's vector type
	switch interface{}(vector).(type) {
	case []float32:
		for i := range m.dim {
			val, err := strconv.ParseFloat(splits[i], 32)
			if err != nil {
				return false, errors.Join(errors.New("Invalid plaintext float"),
					err)
			}
			vector[i] = F(val)
		}
	case []float64:
		for i := range m.dim {
			val, err := strconv.ParseFloat(splits[i], 64)
			if err != nil {
				return false, errors.Join(errors.New("Invalid plaintext float"),
					err)
			}
			vector[i] = F(val)
		}
	default:
		return false, errors.New("Invalid type T when adding plaintext line")
	}
	m.vectors[word] = &FloatVector[F]{scalars: vector}
	return true, nil
}

func (m *FloatModel[F]) FromPlainFile(
	p string, desc bool, _ ...interface{}) error {

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
		if n < 2 {
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

	readMore := true
	for readMore {
		readMore, err = m.readPlainVector(reader)
		if err != nil {
			return err
		}
	}

	return nil
}

// readBinaryVector handles reading vectors when the file and model have the
// same float type
func (m *FloatModel[F]) readBinaryVector(
	br *bufio.Reader) (bool, error) {

	word, err := br.ReadString(' ')
	if err != nil {
		return false, nil
	}
	word = strings.TrimRight(word, " ")

	vector := make([]F, m.dim)
	err = binary.Read(br, binary.LittleEndian, vector)
	if err != nil {
		return false, err
	}

	m.vectors[word] = &FloatVector[F]{scalars: vector}
	return true, nil
}

// castReadFloat32BinaryVector handles reading vectors when the file has
// float32 values but the model doesn't use them
func (m *FloatModel[F]) castReadFloat32BinaryVector(
	br *bufio.Reader) (bool, error) {

	word, err := br.ReadString(' ')
	if err != nil {
		return false, nil
	}
	word = strings.TrimRight(word, " ")

	vector := make([]float32, m.dim)
	err = binary.Read(br, binary.LittleEndian, vector)
	if err != nil {
		return false, err
	}

	vectorf := make([]F, m.dim)
	for i, _ := range vector {
		vectorf[i] = F(vector[i])
	}

	m.vectors[word] = &FloatVector[F]{scalars: vectorf}
	return true, nil
}

// castReadFloat64BinaryVector handles reading vectors when the file has
// float64 values but the model doesn't use them
func (m *FloatModel[F]) castReadFloat64BinaryVector(
	br *bufio.Reader) (bool, error) {

	word, err := br.ReadString(' ')
	if err != nil {
		return false, nil
	}
	word = strings.TrimRight(word, " ")

	vector := make([]float64, m.dim)
	err = binary.Read(br, binary.LittleEndian, vector)
	if err != nil {
		return false, err
	}

	vectorf := make([]F, m.dim)
	for i, _ := range vector {
		vectorf[i] = F(vector[i])
	}

	m.vectors[word] = &FloatVector[F]{scalars: vectorf}
	return true, nil
}

func (m *FloatModel[F]) FromBinaryFile(
	p string, bitSize int, _ ...interface{}) error {

	file, err := os.Open(p)
	defer file.Close()
	if err != nil {
		return err
	}

	reader := bufio.NewReader(file)
	// First line must describe size and dimensions
	var size, dim uint
	n, err := fmt.Fscanln(file, &size, &dim)
	if err != nil {
		return err
	}
	if n < 2 {
		return errors.New("Size and dimensions not found in binary")
	}
	m.dim = dim

	// Since golang doesn't support function overloading and method type
	// parameters, this section looks a little repetitious but the main idea is
	// that when the model vector type matches the binary vector type, we can
	// avoid a cast over every single scalar value on load.
	var f F
	switch any(f).(type) {
	case float32:
		readMore := true
		if bitSize == 64 {
			for readMore {
				readMore, err = m.castReadFloat64BinaryVector(reader)
				if err != nil {
					break
				}
			}
		} else {
			for readMore {
				readMore, err = m.readBinaryVector(reader)
				if err != nil {
					break
				}
			}
		}
	case float64:
		readMore := true
		if bitSize == 64 {
			for readMore {
				readMore, err = m.readBinaryVector(reader)
				if err != nil {
					break
				}
			}
		} else {
			for readMore {
				readMore, err = m.castReadFloat32BinaryVector(reader)
				if err != nil {
					break
				}
			}
		}
	default:
		return errors.New("Loading binary failed. FloatModel should not be " +
			"a type other than a float32 or float64")
	}

	return nil
}
