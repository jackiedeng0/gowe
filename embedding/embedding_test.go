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
