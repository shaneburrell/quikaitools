package gpt2

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Tokenizer is GPT-2 byte-level BPE (vocab.json + merges.txt).
type Tokenizer struct {
	TokenToID map[string]int
	IDToToken []string
	Merges    [][2]string
	Rank      map[[2]string]int
	// UnkID is the fallback id used when a BPE symbol is missing from vocab
	// (broken/incomplete vocab). After byte-level fallback every symbol is in a
	// real GPT-2 vocab; UnkID is kept for source compatibility and equals EOSID
	// when <|endoftext|> is present, otherwise 0.
	UnkID int
	// EOSID is the id of <|endoftext|>, or -1 if that token is absent.
	EOSID int
	cache map[string][]int
}

var (
	byteToUnicode [256]rune
	unicodeToByte map[rune]byte
)

func init() {
	byteToUnicode, unicodeToByte = buildBytesToUnicode()
}

// buildBytesToUnicode is the GPT-2 bytes_to_unicode mapping: printable ranges
// 0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF map to themselves; the remaining 68 bytes
// map to U+0100 upward in byte order.
func buildBytesToUnicode() (enc [256]rune, dec map[rune]byte) {
	dec = make(map[rune]byte, 256)
	n := 0
	for b := 0; b < 256; b++ {
		if (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			enc[b] = rune(b)
			dec[rune(b)] = byte(b)
			continue
		}
		r := rune(0x100 + n)
		enc[b] = r
		dec[r] = byte(b)
		n++
	}
	return enc, dec
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
	tk := &Tokenizer{
		TokenToID: vocab,
		IDToToken: idTo,
		Rank:      map[[2]string]int{},
		EOSID:     -1,
		cache:     map[string][]int{},
	}
	if id, ok := vocab["<|endoftext|>"]; ok {
		tk.EOSID = id
		tk.UnkID = id
	}
	f, err := os.Open(filepath.Join(dir, "merges.txt"))
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	sc := bufio.NewScanner(f)
	i := 0
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
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
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return tk, nil
}

// Encode applies GPT-2 pretokenization, bytes_to_unicode, and ranked BPE.
// Missing symbols (broken vocab) become EOSID if present, otherwise 0.
func (t *Tokenizer) Encode(text string) []int {
	if t == nil || len(t.TokenToID) == 0 {
		return nil
	}
	var ids []int
	for _, piece := range pretokenize(text) {
		ids = append(ids, t.encodePretoken(piece)...)
	}
	return ids
}

func (t *Tokenizer) encodePretoken(piece string) []int {
	if t.cache != nil {
		if cached, ok := t.cache[piece]; ok {
			return cached
		}
	}
	raw := []byte(piece)
	chars := make([]string, len(raw))
	for i, b := range raw {
		chars[i] = string(byteToUnicode[b])
	}
	chars = applyBPE(chars, t.Rank)
	out := make([]int, len(chars))
	fallback := 0
	if t.EOSID >= 0 {
		fallback = t.EOSID
	}
	for i, c := range chars {
		id, ok := t.TokenToID[c]
		if !ok {
			id = fallback
		}
		out[i] = id
	}
	if t.cache == nil {
		t.cache = make(map[string][]int)
	}
	t.cache[piece] = out
	return out
}

func applyBPE(chars []string, rank map[[2]string]int) []string {
	if len(chars) < 2 || len(rank) == 0 {
		return chars
	}
	for {
		best := -1
		bestI := -1
		for i := 0; i < len(chars)-1; i++ {
			pair := [2]string{chars[i], chars[i+1]}
			if r, ok := rank[pair]; ok && (best < 0 || r < best) {
				best = r
				bestI = i
			}
		}
		if bestI < 0 {
			break
		}
		merged := chars[bestI] + chars[bestI+1]
		chars = append(chars[:bestI], append([]string{merged}, chars[bestI+2:]...)...)
	}
	return chars
}

// pretokenize reproduces the GPT-2 regex
// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
// with a rune scanner (RE2 has no (?!\S) lookahead).
func pretokenize(text string) []string {
	if text == "" {
		return nil
	}
	runes := []rune(text)
	var out []string
	i := 0
	n := len(runes)
	for i < n {
		if k := matchContraction(runes[i:]); k > 0 {
			out = append(out, string(runes[i:i+k]))
			i += k
			continue
		}
		if k, ok := matchOptSpaceThen(runes[i:], unicode.IsLetter); ok {
			out = append(out, string(runes[i:i+k]))
			i += k
			continue
		}
		if k, ok := matchOptSpaceThen(runes[i:], unicode.IsNumber); ok {
			out = append(out, string(runes[i:i+k]))
			i += k
			continue
		}
		if k, ok := matchOptSpaceThen(runes[i:], isOther); ok {
			out = append(out, string(runes[i:i+k]))
			i += k
			continue
		}
		if unicode.IsSpace(runes[i]) {
			j := i + 1
			for j < n && unicode.IsSpace(runes[j]) {
				j++
			}
			// \s+(?!\S): a whitespace run followed by a non-space gives up
			// its last whitespace char to the next token.
			if j < n && j > i+1 {
				j--
			}
			out = append(out, string(runes[i:j]))
			i = j
			continue
		}
		out = append(out, string(runes[i]))
		i++
	}
	return out
}

func matchContraction(runes []rune) int {
	if len(runes) < 2 || runes[0] != '\'' {
		return 0
	}
	if len(runes) >= 3 {
		switch string(runes[1:3]) {
		case "re", "ve", "ll":
			return 3
		}
	}
	switch runes[1] {
	case 's', 't', 'm', 'd':
		return 2
	}
	return 0
}

func matchOptSpaceThen(runes []rune, pred func(rune) bool) (int, bool) {
	if len(runes) == 0 {
		return 0, false
	}
	i := 0
	if runes[0] == ' ' {
		i = 1
	}
	if i >= len(runes) || !pred(runes[i]) {
		return 0, false
	}
	i++
	for i < len(runes) && pred(runes[i]) {
		i++
	}
	return i, true
}

func isOther(r rune) bool {
	return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
}

// Decode concatenates token strings, maps each rune through unicode_to_bytes,
// and interprets the result as UTF-8. Inverse of Encode for valid UTF-8.
func (t *Tokenizer) Decode(ids []int) string {
	if t == nil {
		return ""
	}
	var b strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(t.IDToToken) {
			continue
		}
		b.WriteString(t.IDToToken[id])
	}
	s := b.String()
	out := make([]byte, 0, len(s))
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		i += size
		if by, ok := unicodeToByte[r]; ok {
			out = append(out, by)
		}
	}
	return string(out)
}
