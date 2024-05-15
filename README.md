# gowe - Word Embedding Utilities for Go

**gowe** is a Go package for using word embeddings.

## Motivation
There are existing packages for word embeddings in Go that have inspired this package, but most have not seen updates for a while:
- [ynqa/wego](https://github.com/ynqa/wego)
- [sajari/word2vec](https://github.com/sajari/word2vec)

This motivates the creation of a new package built for modern Go (1.22+).

## API

Import using:

```go
import "github.com/jackiedeng0/gowe"
```

To Use:
```go
model, err := LoadFromPlainFile("glove.6B.50d.txt")
// You can retrieve this model at https://github.com/stanfordnlp/GloVe/

// Get the vector embedding for a word
fmt.Log(model.Vector("cat"))
// {[0.45281 -0.50108 -0.53714 -0.015697 0.22191 ... ]}

// Get the similarity (cosine) between two words
fmt.Logf("%0.3f\n", model.Similarity("cat", "dog"))
// 0.922

// Within a list of words, exhaustively search and rank the N most similar words
words := []string{"dog", "apple", "lincoln", "whisker", "road", "cheetah"}
nearest, err := model.NNearestIn("cat", words, 3)
fmt.Log(nearest)
// [dog cheetah apple]
```


## Status
- [x] Load plaintext model files as float64 embedding models
- [x] Float and Int generic vector types
- [x] Quantization and Dequantization
- [ ] Loading binary model files
- [ ] Loading models as any vector type
