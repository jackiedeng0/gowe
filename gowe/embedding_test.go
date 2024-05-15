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
	"log"
	"os"
	"strings"
	"testing"
)

var model *Model
var testVocab []string

func TestMain(m *testing.M) {
	// Glove model retrieved from https://github.com/stanfordnlp/GloVe/
	// Download model and place in this directory if you wish to run this test
	var err error
	model, err = LoadFromPlainFile("glove.6B.50d.txt")
	if err != nil {
		log.Fatal(err)
	}

	data, err := os.ReadFile("test_vocabulary.txt")
	if err != nil {
		log.Fatal(err)
	}
	testVocab = strings.Split(string(data), "\n")

	os.Exit(m.Run())
}

func TestEmbedding(t *testing.T) {
	t.Logf("model has %d dimensions and a vocabulary of %d words",
		model.Dimensions(), model.VocabularySize())

	t.Logf("The encoding for \"cat\" is %v", model.Vector("cat"))
	t.Logf("The encoding for \"_not_a_word\" is %v", model.Vector("not_a_word"))
	t.Logf("The similarity \"cat\" and \"dog\" is %0.3f",
		model.Similarity("cat", "dog"))
	t.Logf("The similarity \"cat\" and \"lincoln\" is %0.3f",
		model.Similarity("cat", "lincoln"))

	words := []string{"dog", "apple", "lincoln", "whisker", "road", "cheetah"}
	t.Log(model.NNearestIn("cat", words, 3))
}

func BenchmarkNNearest5in10(b *testing.B) {
	for i := 0; i < b.N; i++ {
		model.NNearestIn("cat", testVocab[:10], 5)
	}
}

func BenchmarkNNearest5in100(b *testing.B) {
	for i := 0; i < b.N; i++ {
		model.NNearestIn("cat", testVocab[:100], 5)
	}
}

func BenchmarkNNearest5in1000(b *testing.B) {
	for i := 0; i < b.N; i++ {
		model.NNearestIn("cat", testVocab[:1000], 5)
	}
}
