package gpt2

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadDirRejectsWrongTensorSize(t *testing.T) {
	dir := t.TempDir()
	base := NewRandom(Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 8, NInner: 16}, 1)
	if err := WriteDir(dir, base); err != nil {
		t.Fatal(err)
	}
	cfg := Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 1000, NInner: 16}
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = LoadDir(dir)
	if err == nil || !strings.Contains(err.Error(), "wte.weight") {
		t.Fatalf("want wte length error, got %v", err)
	}
}

func TestLoadDirRejectsMissingPositions(t *testing.T) {
	dir := t.TempDir()
	base := NewRandom(Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 8, NInner: 16}, 1)
	if err := WriteDir(dir, base); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(map[string]int{"n_embd": 8, "n_head": 2, "n_layer": 1, "vocab_size": 8})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = LoadDir(dir)
	if err == nil || !strings.Contains(err.Error(), "incomplete config") {
		t.Fatalf("want incomplete config, got %v", err)
	}
}
