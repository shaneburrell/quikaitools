package zoo

import (
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"
)

func TestParseAndFilter(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("embed.yaml", `
id: demo-embed
name: Demo Embed
task: embed
formats: [onnx]
license: apache-2.0
machines:
  v100: works
  halo: cpu_only
  mac: works
engine:
  v100: hugot-ort-cuda
`)
	write("llm.yaml", `
id: demo-llm
name: Demo LLM
task: generate
formats: [gguf]
license: apache-2.0
machines:
  v100: works
  halo: works
  mac: works
`)
	cat, err := LoadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(cat.Models) != 2 {
		t.Fatalf("len=%d", len(cat.Models))
	}
	embeds := cat.Filter("embed", "v100")
	if len(embeds) != 1 || embeds[0].ID != "demo-embed" {
		t.Fatalf("filter embed/v100 = %+v", embeds)
	}
	if embeds[0].EngineOn("v100") != "hugot-ort-cuda" {
		t.Fatalf("engine = %s", embeds[0].EngineOn("v100"))
	}
}

func TestLoadFSAndMissingID(t *testing.T) {
	fsys := fstest.MapFS{
		"ok.yaml": &fstest.MapFile{Data: []byte("id: x\ntask: embed\n")},
	}
	cat, err := LoadFS(fsys)
	if err != nil {
		t.Fatal(err)
	}
	if len(cat.Models) != 1 {
		t.Fatalf("len=%d", len(cat.Models))
	}
	if _, err := parseModel([]byte("task: embed\n"), "bad.yaml"); err == nil {
		t.Fatal("expected missing id")
	}
}

func TestLoadDirNestedModels(t *testing.T) {
	root := t.TempDir()
	models := filepath.Join(root, "models")
	if err := os.Mkdir(models, 0o755); err != nil {
		t.Fatal(err)
	}
	body := "id: nested\ntask: classify\n"
	if err := os.WriteFile(filepath.Join(models, "n.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cat, err := LoadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(cat.Models) != 1 || cat.Models[0].ID != "nested" {
		t.Fatalf("got %+v", cat.Models)
	}
}

func TestStatusOnDefault(t *testing.T) {
	var m Model
	if m.StatusOn("v100") != StatusUntested {
		t.Fatalf("nil machines: %s", m.StatusOn("v100"))
	}
	if m.EngineOn("v100") != "" {
		t.Fatal("nil engine")
	}
}

func TestGetAndFilesAndSource(t *testing.T) {
	cat := Catalog{Models: []Model{
		{ID: "tiny-random-gpt2", Source: "https://huggingface.co/hf-internal-testing/tiny-random-gpt2", Formats: []string{"safetensors"}},
		{ID: "clip", Source: "openai/clip-vit-base-patch32", Formats: []string{"onnx"}},
	}}
	m, ok := cat.Get("tiny-random-gpt2")
	if !ok || RepoFromSource(m.Source) != "hf-internal-testing/tiny-random-gpt2" {
		t.Fatalf("%+v", m)
	}
	if r := RepoFromSource("org/name"); r != "org/name" {
		t.Fatalf("%s", r)
	}
	files := FilesForModel(m)
	if len(files) < 3 {
		t.Fatalf("safetensors files=%v", files)
	}
	onnx := FilesForModel(Model{Formats: []string{"onnx"}})
	if len(onnx) < 2 {
		t.Fatalf("onnx=%v", onnx)
	}
	gguf := FilesForModel(Model{Formats: []string{"gguf"}})
	if len(gguf) == 0 {
		t.Fatal("gguf")
	}
}
