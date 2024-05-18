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

Load plaintext file to a float 32 model:
```go
model := newFloatModel[float32]()
err := model.FromPlainFile("glove.6B.50d.txt", false)
// You can retrieve this model at https://github.com/stanfordnlp/GloVe/
// 'false' because the file doesn't have a "<size> <dim>" description

// Get the vector embedding for a word
fmt.Println(model.Vector("cat"))
// [1.45281 -0.50108 -0.53714 -0.015697 0.22191 ... ]

// Get the similarity (cosine) between two words
fmt.Printf("%0.3f\n", model.Similarity("cat", "dog"))
// 0.922

// Within a list of words, exhaustively search and rank the N most similar words
words := []string{"dog", "apple", "lincoln", "whisker", "road", "cheetah"}
nearest, err := model.NNearestIn("cat", words, 3)
fmt.Println(nearest)
// [dog cheetah apple]
```

Load plaintext file to a quantized int model (int8, int16, int32 supported):
```go
model := newIntModel[int16]()
err := model.FromPlainFile("glove.6B.50d.txt", false, 5.0)
// Requires an additional float64 argument for the maximum magnitude of any
// scalar value - in this case, it was 5.0. For a normalized model, this would
// be 1.0
```

Load binary file to float and int models respectively:
```go
floatModel := newFloatModel[float32]()
err := model.FromBinaryFile("model.bin", 32)
// Description is always provided, so we just need to specify the bitSize of
// floating points in the file

intModel := newIntModel[int8]()
err := model.FromBinaryFile("model.bin", 32, 2.0)
```

## Status
- [x] Load plaintext model files as float64 embedding models
- [x] Float and Int generic vector types
- [x] Quantization and Dequantization
- [x] Loading models as any vector type
- [x] Loading binary model files
