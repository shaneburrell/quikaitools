package gpt2

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"
)

// Tokenizer is a small GPT-2 BPE (vocab.json + merges.txt).
type Tokenizer struct {
	TokenToID map[string]int
	IDToToken []string
	Merges    [][2]string
	Rank      map[[2]string]int
	UnkID     int
}

// LoadTokenizer reads vocab.json and merges.txt from a Hub folder.
func LoadTokenizer(dir string) (*Tokenizer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "vocab.json"))
	if err != nil {
		return nil, err
	}
	var vocab map[string]int
	if err := json.Unmarshal(raw, &vocab); err != nil {
		return nil, err
	}
	maxID := 0
	for _, id := range vocab {
		if id > maxID {
			maxID = id
		}
	}
	idTo := make([]string, maxID+1)
	for tok, id := range vocab {
		if id >= 0 && id < len(idTo) {
			idTo[id] = tok
		}
	}
	tk := &Tokenizer{TokenToID: vocab, IDToToken: idTo, Rank: map[[2]string]int{}, UnkID: 0}
	if id, ok := vocab["<|endoftext|>"]; ok {
		tk.UnkID = id
	}
	f, err := os.Open(filepath.Join(dir, "merges.txt"))
	if err != nil {
		return tk, nil
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	i := 0
	for sc.Scan() {
		line := sc.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, " ")
		if len(parts) != 2 {
			continue
		}
		pair := [2]string{parts[0], parts[1]}
		tk.Merges = append(tk.Merges, pair)
		tk.Rank[pair] = i
		i++
	}
	return tk, sc.Err()
}

// Encode applies byte-level BPE. Unknown pieces become UnkID.
func (t *Tokenizer) Encode(text string) []int {
	if t == nil || len(t.TokenToID) == 0 {
		return nil
	}
	var ids []int
	for _, word := range splitWords(text) {
		ids = append(ids, t.encodeWord(word)...)
	}
	return ids
}

func (t *Tokenizer) encodeWord(word string) []int {
	chars := strings.Split(word, "")
	if len(chars) == 0 {
		return nil
	}
	for {
		best := -1
		bestI := -1
		for i := 0; i < len(chars)-1; i++ {
			pair := [2]string{chars[i], chars[i+1]}
			if r, ok := t.Rank[pair]; ok && (best < 0 || r < best) {
				best = r
				bestI = i
			}
		}
		if bestI < 0 {
			break
		}
		chars = append(chars[:bestI], append([]string{chars[bestI] + chars[bestI+1]}, chars[bestI+2:]...)...)
	}
	out := make([]int, len(chars))
	for i, c := range chars {
		id, ok := t.TokenToID[c]
		if !ok {
			id = t.UnkID
		}
		out[i] = id
	}
	return out
}

func splitWords(text string) []string {
	// GPT-2 uses Ġ for leading space. Keep it simple and local.
	var words []string
	var b strings.Builder
	first := true
	for len(text) > 0 {
		r, size := utf8.DecodeRuneInString(text)
		text = text[size:]
		if r == ' ' || r == '\n' || r == '\t' {
			if b.Len() > 0 {
				words = append(words, b.String())
				b.Reset()
			}
			first = false
			// next word starts with Ġ
			if r == ' ' {
				b.WriteString("Ġ")
			}
			continue
		}
		if first {
			first = false
		}
		b.WriteRune(r)
	}
	if b.Len() > 0 {
		words = append(words, b.String())
	}
	return words
}

// Decode joins tokens, turning Ġ into space.
func (t *Tokenizer) Decode(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(t.IDToToken) {
			continue
		}
		tok := t.IDToToken[id]
		tok = strings.ReplaceAll(tok, "Ġ", " ")
		b.WriteString(tok)
	}
	return b.String()
}
