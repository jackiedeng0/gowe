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

func TestEmbedding(t *testing.T) {
	// Glove model retrieved from https://github.com/stanfordnlp/GloVe/
	// Download model and place in this directory if you wish to run this test
	model, err := LoadFromPlainFile("glove.6B.50d.txt")
	if err != nil {
		t.Error(err)
	}
	t.Log(model.Vector("a"))
	t.Logf("model has %d dimensions and a vocabulary of %d words",
		model.Dimensions(), model.VocabularySize())
}
