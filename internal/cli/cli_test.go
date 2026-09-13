package cli

import (
	"bytes"
	"encoding/json"
	"image/color"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/audio"
	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/vision"
)

func TestVersionAndHelp(t *testing.T) {
	var out bytes.Buffer
	if code := Main([]string{"quikaitools", "version"}, &out, &out); code != 0 {
		t.Fatalf("code=%d out=%s", code, out.String())
	}
	if !strings.Contains(out.String(), Version) {
		t.Fatalf("version output: %s", out.String())
	}
	out.Reset()
	if code := Main([]string{"quikaitools", "--version"}, &out, &out); code != 0 {
		t.Fatalf("--version code=%d out=%s", code, out.String())
	}
	if !strings.Contains(out.String(), Version) {
		t.Fatalf("--version output: %s", out.String())
	}
	out.Reset()
	if code := Main([]string{"quikaitools", "help"}, &out, &out); code != 0 {
		t.Fatalf("help code=%d", code)
	}
	if !strings.Contains(out.String(), "doctor") || !strings.Contains(out.String(), "train") || !strings.Contains(out.String(), "export") {
		t.Fatalf("help: %s", out.String())
	}
}

func TestDoctorProfile(t *testing.T) {
	var out, err bytes.Buffer
	code := Main([]string{"quikaitools", "doctor", "--profile", "v100"}, &out, &err)
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, err.String())
	}
	if !strings.Contains(out.String(), "gomlx-xla-cuda") {
		t.Fatalf("doctor: %s", out.String())
	}
	if !strings.Contains(out.String(), "binding:") {
		t.Fatalf("expected binding note: %s", out.String())
	}
	if !strings.Contains(out.String(), "FP32") {
		t.Fatalf("expected fp16 warning: %s", out.String())
	}
}

func TestCatalogEmbedded(t *testing.T) {
	var out, err bytes.Buffer
	code := Main([]string{"quikaitools", "catalog", "--machine", "halo", "--task", "generate"}, &out, &err)
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, err.String())
	}
	if !strings.Contains(out.String(), "gemma3-270m-it") {
		t.Fatalf("catalog: %s", out.String())
	}
}

func TestUnknownCommand(t *testing.T) {
	var out, err bytes.Buffer
	if code := Main([]string{"quikaitools", "nope"}, &out, &err); code != 2 {
		t.Fatalf("code=%d", code)
	}
}

func TestPullAndTrainHelp(t *testing.T) {
	var out, err bytes.Buffer
	if code := Main([]string{"quikaitools", "pull", "--help"}, &out, &err); code != 0 {
		t.Fatalf("pull help %d", code)
	}
	out.Reset()
	err.Reset()
	if code := Main([]string{"quikaitools", "train"}, &out, &err); code != 0 {
		t.Fatalf("train help code=%d", code)
	}
	if !strings.Contains(out.String(), "lora") {
		t.Fatalf("train help: %s", out.String())
	}
}

func TestDoctorHelpAndBadFlag(t *testing.T) {
	var out, err bytes.Buffer
	if code := Main([]string{"quikaitools", "doctor", "--help"}, &out, &err); code != 0 {
		t.Fatalf("help code=%d", code)
	}
	out.Reset()
	err.Reset()
	if code := Main([]string{"quikaitools", "doctor", "--nope"}, &out, &err); code != 2 {
		t.Fatalf("bad flag code=%d", code)
	}
}

func TestGenerateEmbedTranscribeHelp(t *testing.T) {
	var out, errb bytes.Buffer
	for _, args := range [][]string{
		{"quikaitools", "generate", "--help"},
		{"quikaitools", "embed", "--help"},
		{"quikaitools", "transcribe", "--help"},
		{"quikaitools", "doctor", "--strict", "--profile", "cpu"},
	} {
		out.Reset()
		errb.Reset()
		code := Main(args, &out, &errb)
		if args[1] == "doctor" {
			if !strings.Contains(out.String()+errb.String(), "strict") && code > 1 {
				t.Fatalf("%v code=%d out=%s err=%s", args, code, out.String(), errb.String())
			}
			continue
		}
		if code != 0 {
			t.Fatalf("%v code=%d err=%s", args, code, errb.String())
		}
	}
}

func TestGenerateWithAdapterRoundTrip(t *testing.T) {
	dir := t.TempDir()
	mustWriteTiny(t, dir)
	data := filepath.Join(dir, "d.txt")
	_ = os.WriteFile(data, []byte("aaaaaaa bbbbbbb ccccccc"), 0o644)
	adapter := filepath.Join(dir, "ad")
	var sink bytes.Buffer
	code := Main([]string{"quikaitools", "train", "lora", "--model", dir, "--data", data, "--steps", "2", "--rank", "2", "--out", adapter, "--profile", "cpu"}, &sink, &sink)
	if code != 0 {
		t.Fatalf("train code=%d out=%s", code, sink.String())
	}
	var out, errb bytes.Buffer
	code = Main([]string{"quikaitools", "generate", "--model", dir, "--adapter", adapter, "--prompt", "aa", "--tokens", "2"}, &out, &errb)
	if code != 0 {
		t.Fatalf("generate code=%d err=%s", code, errb.String())
	}
	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "embed", "--model", dir, "--text", "aa"}, &out, &errb)
	if code != 0 {
		t.Fatalf("embed code=%d err=%s", code, errb.String())
	}
	img := filepath.Join(dir, "x.png")
	if err := vision.SolidPNG(img, color.RGBA{R: 255, A: 255}, 32, 32); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "embed", "--vision", "--model", dir, "--image", img}, &out, &errb)
	if code != 0 {
		t.Fatalf("vision code=%d err=%s", code, errb.String())
	}
	if strings.Contains(out.String(), "NaN") {
		t.Fatalf("nan embed: %s", out.String())
	}
	wav := filepath.Join(dir, "t.wav")
	if err := audio.WriteToneWAV(wav, 16000, 4000, 440); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "transcribe", "--model", dir, "--audio", wav}, &out, &errb)
	if code != 0 {
		t.Fatalf("asr code=%d err=%s", code, errb.String())
	}
}

func mustWriteTiny(t *testing.T, dir string) {
	t.Helper()
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 32, VocabSize: 32, NInner: 16}, 11)
	if err := gpt2.WriteDir(dir, base); err != nil {
		t.Fatal(err)
	}
}

func TestExportMergeAndValidateSFT(t *testing.T) {
	dir := t.TempDir()
	mustWriteTiny(t, dir)
	data := filepath.Join(dir, "train.jsonl")
	row := `{"messages":[{"role":"system","content":"s"},{"role":"user","content":"hello world tokens"},{"role":"assistant","content":"hi there friend"}]}` + "\n"
	if err := os.WriteFile(data, []byte(row), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := Main([]string{"quikaitools", "validate-sft", "--path", data}, &out, &errb); code != 0 {
		t.Fatalf("validate code=%d err=%s", code, errb.String())
	}
	adapter := filepath.Join(dir, "ad")
	out.Reset()
	errb.Reset()
	code := Main([]string{"quikaitools", "train", "lora", "--smoke", "--model", dir, "--data", data, "--steps", "2", "--rank", "2", "--out", adapter, "--profile", "cpu"}, &out, &errb)
	if code != 0 {
		t.Fatalf("train code=%d err=%s out=%s", code, errb.String(), out.String())
	}
	merged := filepath.Join(dir, "merged")
	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "export", "merge", "--model", dir, "--adapter", adapter, "--out", merged}, &out, &errb)
	if code != 0 {
		t.Fatalf("merge code=%d err=%s", code, errb.String())
	}
	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "export", "gguf", "--model", merged, "--out", filepath.Join(dir, "m.gguf")}, &out, &errb)
	if code != 0 {
		t.Fatalf("gguf code=%d err=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "SKIP") && !strings.Contains(out.String(), "gguf") {
		t.Fatalf("gguf out=%s", out.String())
	}
	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "generate", "--model", dir, "--template", "chatml", "--messages", `[{"role":"user","content":"aa"}]`, "--tokens", "2"}, &out, &errb)
	if code != 0 {
		t.Fatalf("generate template code=%d err=%s", code, errb.String())
	}
}

func TestExportTrailingFlagsNoPanic(t *testing.T) {
	for _, args := range [][]string{
		{"quikaitools", "export", "merge", "--model"},
		{"quikaitools", "export", "gguf", "--out"},
		{"quikaitools", "export", "modelfile", "--gguf"},
	} {
		var out, errb bytes.Buffer
		code := Main(args, &out, &errb)
		if code != 2 {
			t.Fatalf("%v code=%d err=%s", args, code, errb.String())
		}
		if !strings.Contains(errb.String(), "needs a value") {
			t.Fatalf("%v stderr=%s", args, errb.String())
		}
	}
}

func TestTrainGenerateValidation(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{"steps", []string{"quikaitools", "train", "lora", "--model", "m", "--data", "d", "--steps", "0"}},
		{"rank", []string{"quikaitools", "train", "lora", "--model", "m", "--data", "d", "--rank", "0"}},
		{"accum", []string{"quikaitools", "train", "lora", "--model", "m", "--data", "d", "--accum", "0"}},
		{"eval-every", []string{"quikaitools", "train", "lora", "--model", "m", "--data", "d", "--eval-every", "0"}},
		{"lr", []string{"quikaitools", "train", "lora", "--model", "m", "--data", "d", "--lr", "0"}},
		{"seq", []string{"quikaitools", "train", "lora", "--model", "m", "--data", "d", "--seq", "0"}},
		{"tokens", []string{"quikaitools", "generate", "--model", "m", "--tokens", "0"}},
		{"template", []string{"quikaitools", "generate", "--model", "m", "--template", "chatml"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var out, errb bytes.Buffer
			code := Main(tc.args, &out, &errb)
			if code != 2 {
				t.Fatalf("code=%d err=%s", code, errb.String())
			}
			if errb.Len() == 0 {
				t.Fatal("expected stderr message")
			}
		})
	}
}

func TestTrainStepsAndSmokeOrder(t *testing.T) {
	dir := t.TempDir()
	mustWriteTiny(t, dir)
	data := filepath.Join(dir, "d.txt")
	if err := os.WriteFile(data, []byte("aaaaaaa bbbbbbb ccccccc ddddddd"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := Main([]string{"quikaitools", "train", "lora", "--model", dir, "--data", data, "--steps", "4", "--rank", "2", "--accum", "2", "--out", filepath.Join(dir, "ad1"), "--profile", "cpu"}, &out, &errb)
	if code != 0 {
		t.Fatalf("train code=%d err=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "lora steps=4 optimizer_steps=") {
		t.Fatalf("expected steps and optimizer_steps: %s", out.String())
	}

	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "train", "lora", "--model", dir, "--data", data, "--steps", "100", "--rank", "2", "--smoke", "--out", filepath.Join(dir, "ad2"), "--profile", "cpu"}, &out, &errb)
	if code != 0 {
		t.Fatalf("smoke code=%d err=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "lora steps=20 optimizer_steps=") {
		t.Fatalf("smoke should force steps=20: %s", out.String())
	}
	if !strings.Contains(out.String(), "accum=4") {
		t.Fatalf("smoke should force accum=4: %s", out.String())
	}
}

func TestPullOfflineCacheAndCatalogID(t *testing.T) {
	src := t.TempDir()
	mustWriteTiny(t, src)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		const marker = "/resolve/main/"
		idx := strings.Index(r.URL.Path, marker)
		if idx < 0 {
			http.NotFound(w, r)
			return
		}
		name := r.URL.Path[idx+len(marker):]
		if name == "" || strings.Contains(name, "..") {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, filepath.Join(src, filepath.FromSlash(name)))
	}))
	t.Cleanup(srv.Close)

	cache := t.TempDir()
	t.Setenv("HF_ENDPOINT", srv.URL)
	t.Setenv("QUIKAITOOLS_CACHE", cache)
	t.Setenv("QUIKAITOOLS_CATALOG", "")

	var out, errb bytes.Buffer
	code := Main([]string{"quikaitools", "pull", "some-org/some-model"}, &out, &errb)
	if code != 0 {
		t.Fatalf("pull code=%d err=%s out=%s", code, errb.String(), out.String())
	}
	got := filepath.Join(cache, "models", "some-org", "some-model")
	for _, name := range []string{"config.json", "vocab.json", "merges.txt", "model.safetensors"} {
		if _, err := os.Stat(filepath.Join(got, name)); err != nil {
			t.Fatalf("missing %s: %v", name, err)
		}
	}

	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "pull", "some-org/some-model"}, &out, &errb)
	if code != 0 {
		t.Fatalf("cache-hit pull code=%d err=%s out=%s", code, errb.String(), out.String())
	}

	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "pull", "tiny-random-gpt2"}, &out, &errb)
	if code != 0 {
		t.Fatalf("catalog pull code=%d err=%s out=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "hf-internal-testing/tiny-random-gpt2") {
		t.Fatalf("expected catalog resolve: %s", out.String())
	}
	catDir := filepath.Join(cache, "models", "hf-internal-testing", "tiny-random-gpt2")
	if _, err := os.Stat(filepath.Join(catDir, "model.safetensors")); err != nil {
		t.Fatalf("catalog pull missing weights: %v", err)
	}
}

func TestTrainQLoRAEndToEnd(t *testing.T) {
	dir := t.TempDir()
	mustWriteTiny(t, dir)
	data := filepath.Join(dir, "d.txt")
	if err := os.WriteFile(data, []byte("aaaaaaa bbbbbbb ccccccc ddddddd"), 0o644); err != nil {
		t.Fatal(err)
	}
	outDir := filepath.Join(dir, "qlora-out")
	var out, errb bytes.Buffer
	code := Main([]string{"quikaitools", "train", "qlora", "--model", dir, "--data", data, "--steps", "3", "--rank", "2", "--out", outDir, "--profile", "cpu"}, &out, &errb)
	if code != 0 {
		t.Fatalf("qlora code=%d err=%s out=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "qlora") {
		t.Fatalf("expected qlora in output: %s", out.String())
	}
	if _, err := os.Stat(filepath.Join(outDir, "adapter.json")); err != nil {
		t.Fatalf("adapter.json: %v", err)
	}
}

func TestTrainResumeStepAndOptimizer(t *testing.T) {
	dir := t.TempDir()
	mustWriteTiny(t, dir)
	data := filepath.Join(dir, "d.txt")
	if err := os.WriteFile(data, []byte("aaaaaaa bbbbbbb ccccccc ddddddd eeeeeee"), 0o644); err != nil {
		t.Fatal(err)
	}
	a := filepath.Join(dir, "A")
	b := filepath.Join(dir, "B")
	var out, errb bytes.Buffer
	code := Main([]string{"quikaitools", "train", "lora", "--model", dir, "--data", data, "--steps", "4", "--rank", "2", "--out", a, "--profile", "cpu"}, &out, &errb)
	if code != 0 {
		t.Fatalf("train A code=%d err=%s out=%s", code, errb.String(), out.String())
	}
	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "train", "lora", "--model", dir, "--data", data, "--steps", "2", "--rank", "2", "--resume", a, "--out", b, "--profile", "cpu"}, &out, &errb)
	if code != 0 {
		t.Fatalf("resume B code=%d err=%s out=%s", code, errb.String(), out.String())
	}
	raw, err := os.ReadFile(filepath.Join(b, "meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	var meta struct {
		Step int `json:"step"`
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	if meta.Step != 6 {
		t.Fatalf("meta.step=%d want 6 (%s)", meta.Step, raw)
	}
	if _, err := os.Stat(filepath.Join(b, "optimizer.json")); err != nil {
		t.Fatalf("optimizer.json: %v", err)
	}
}

func TestTrainResumeKeepsRankAndLR(t *testing.T) {
	dir := t.TempDir()
	mustWriteTiny(t, dir)
	data := filepath.Join(dir, "d.txt")
	if err := os.WriteFile(data, []byte("aaaaaaa bbbbbbb ccccccc ddddddd eeeeeee"), 0o644); err != nil {
		t.Fatal(err)
	}
	a := filepath.Join(dir, "A")
	b := filepath.Join(dir, "B")
	var out, errb bytes.Buffer
	code := Main([]string{"quikaitools", "train", "lora", "--model", dir, "--data", data, "--steps", "2", "--rank", "8", "--lr", "1e-4", "--out", a, "--profile", "cpu"}, &out, &errb)
	if code != 0 {
		t.Fatalf("train A code=%d err=%s out=%s", code, errb.String(), out.String())
	}
	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "train", "lora", "--model", dir, "--data", data, "--steps", "2", "--resume", a, "--out", b, "--profile", "cpu"}, &out, &errb)
	if code != 0 {
		t.Fatalf("resume without rank/lr code=%d err=%s out=%s", code, errb.String(), out.String())
	}
	raw, err := os.ReadFile(filepath.Join(b, "adapter.json"))
	if err != nil {
		t.Fatal(err)
	}
	var file struct {
		LoRA struct {
			Rank int     `json:"rank"`
			LR   float64 `json:"lr"`
		} `json:"lora"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatal(err)
	}
	if file.LoRA.Rank != 8 {
		t.Fatalf("resume rank=%d want 8", file.LoRA.Rank)
	}
	if file.LoRA.LR != 1e-4 {
		t.Fatalf("resume lr=%g want 1e-4", file.LoRA.LR)
	}
	out.Reset()
	errb.Reset()
	c := filepath.Join(dir, "C")
	code = Main([]string{"quikaitools", "train", "lora", "--model", dir, "--data", data, "--steps", "1", "--rank", "4", "--resume", a, "--out", c, "--profile", "cpu"}, &out, &errb)
	if code == 0 {
		t.Fatal("expected rank mismatch on resume --rank 4")
	}
	if !strings.Contains(errb.String(), "rank") {
		t.Fatalf("want rank error, got %s", errb.String())
	}
}

func TestTrainMaskPromptJSONL(t *testing.T) {
	dir := t.TempDir()
	mustWriteTiny(t, dir)
	data := filepath.Join(dir, "sft.jsonl")
	row := `{"messages":[{"role":"system","content":"be helpful always"},{"role":"user","content":"hello world tokens please"},{"role":"assistant","content":"hi there friend reply now"}]}` + "\n"
	if err := os.WriteFile(data, []byte(row+row+row), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := Main([]string{"quikaitools", "train", "lora", "--model", dir, "--data", data, "--steps", "2", "--rank", "2", "--mask-prompt", "--out", filepath.Join(dir, "ad"), "--profile", "cpu"}, &out, &errb)
	if code != 0 {
		t.Fatalf("mask-prompt code=%d err=%s out=%s", code, errb.String(), out.String())
	}
	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "train", "lora", "--model", dir, "--data", data, "--steps", "2", "--rank", "2", "--mask-prompt=false", "--out", filepath.Join(dir, "ad2"), "--profile", "cpu"}, &out, &errb)
	if code != 0 {
		t.Fatalf("mask-prompt=false code=%d err=%s out=%s", code, errb.String(), out.String())
	}
}

func TestTrainSeedDeterministic(t *testing.T) {
	dir := t.TempDir()
	mustWriteTiny(t, dir)
	data := filepath.Join(dir, "d.txt")
	if err := os.WriteFile(data, []byte("abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz"), 0o644); err != nil {
		t.Fatal(err)
	}
	run := func(outDir string) string {
		t.Helper()
		var out, errb bytes.Buffer
		code := Main([]string{"quikaitools", "train", "lora", "--model", dir, "--data", data, "--steps", "4", "--rank", "2", "--seed", "42", "--seq", "8", "--out", outDir, "--profile", "cpu"}, &out, &errb)
		if code != 0 {
			t.Fatalf("seed train code=%d err=%s out=%s", code, errb.String(), out.String())
		}
		for _, line := range strings.Split(out.String(), "\n") {
			if strings.Contains(line, "loss_first=") {
				return strings.TrimSpace(line)
			}
		}
		t.Fatalf("no loss line: %s", out.String())
		return ""
	}
	a := run(filepath.Join(dir, "a"))
	b := run(filepath.Join(dir, "b"))
	if a != b {
		t.Fatalf("same seed losses differ:\n%s\n%s", a, b)
	}
}

func TestGenerateSamplingFlags(t *testing.T) {
	dir := t.TempDir()
	mustWriteTiny(t, dir)
	var out, errb bytes.Buffer
	code := Main([]string{"quikaitools", "generate", "--model", dir, "--prompt", "aa", "--temperature", "0.8", "--seed", "1", "--tokens", "5"}, &out, &errb)
	if code != 0 {
		t.Fatalf("sample code=%d err=%s", code, errb.String())
	}
	if strings.TrimSpace(out.String()) == "" {
		t.Fatal("expected generated text")
	}

	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "generate", "--model", dir, "--temperature", "-1"}, &out, &errb)
	if code != 2 {
		t.Fatalf("neg temperature code=%d err=%s", code, errb.String())
	}

	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "generate", "--model", dir, "--prompt", "aa", "--stop-eos", "--tokens", "3"}, &out, &errb)
	if code != 0 {
		t.Fatalf("stop-eos code=%d err=%s", code, errb.String())
	}
}

func TestGenerateGGUFFakeLlama(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake llama-cli shell script")
	}
	binDir := t.TempDir()
	script := filepath.Join(binDir, "llama-cli")
	if err := os.WriteFile(script, []byte("#!/bin/sh\necho fake-gguf-output\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", binDir)
	t.Setenv("QUIKAITOOLS_LLAMA", script)
	gguf := filepath.Join(binDir, "empty.gguf")
	if err := os.WriteFile(gguf, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := Main([]string{"quikaitools", "generate", "--gguf", gguf, "--prompt", "hi", "--tokens", "2"}, &out, &errb)
	if code != 0 {
		t.Fatalf("gguf code=%d err=%s out=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "fake-gguf-output") {
		t.Fatalf("gguf out=%s", out.String())
	}

	slow := filepath.Join(binDir, "slow-llama")
	// PATH is restricted to binDir above, so use an absolute path for sleep;
	// otherwise `sleep` is not found and the script echoes immediately.
	if err := os.WriteFile(slow, []byte("#!/bin/sh\n/bin/sleep 2\necho should-not-see\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("QUIKAITOOLS_LLAMA", slow)
	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "generate", "--gguf", gguf, "--timeout", "1ms"}, &out, &errb)
	if code != 1 {
		t.Fatalf("timeout code=%d err=%s out=%s", code, errb.String(), out.String())
	}
	msg := strings.ToLower(errb.String())
	if !strings.Contains(msg, "timed out") && !strings.Contains(msg, "timeout") && !strings.Contains(msg, "deadline") {
		t.Fatalf("expected timeout message: %s", errb.String())
	}
}

func TestExportModelfileForce(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "Modelfile")
	var out, errb bytes.Buffer
	code := Main([]string{"quikaitools", "export", "modelfile", "--gguf", "m.gguf", "--out", outPath}, &out, &errb)
	if code != 0 {
		t.Fatalf("modelfile code=%d err=%s", code, errb.String())
	}
	body, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), "FROM") {
		t.Fatalf("modelfile: %s", body)
	}

	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "export", "modelfile", "--gguf", "m.gguf", "--out", outPath}, &out, &errb)
	if code != 1 {
		t.Fatalf("overwrite without --force code=%d err=%s", code, errb.String())
	}

	out.Reset()
	errb.Reset()
	code = Main([]string{"quikaitools", "export", "modelfile", "--gguf", "m.gguf", "--out", outPath, "--force"}, &out, &errb)
	if code != 0 {
		t.Fatalf("--force code=%d err=%s", code, errb.String())
	}
}

func TestValidateSFTErrors(t *testing.T) {
	var out, errb bytes.Buffer
	if code := Main([]string{"quikaitools", "validate-sft"}, &out, &errb); code != 2 {
		t.Fatalf("missing --path code=%d err=%s", code, errb.String())
	}

	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.jsonl")
	// A complete non-object JSON value (decoder consumes it) plus a row missing role.
	// A raw non-JSON token like "not-json" leaves encoding/json.Decoder unadvanced
	// and ValidateMessagesJSONL loops forever (train is outside this change set).
	payload := "123\n" + `{"messages":[{"content":"no role here"},{"role":"user","content":"x"}]}` + "\n"
	if err := os.WriteFile(bad, []byte(payload), 0o644); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errb.Reset()
	code := Main([]string{"quikaitools", "validate-sft", "--path", bad}, &out, &errb)
	if code != 1 {
		t.Fatalf("invalid jsonl code=%d out=%s err=%s", code, out.String(), errb.String())
	}
	combined := out.String() + errb.String()
	if !strings.Contains(combined, "INVALID") && !strings.Contains(combined, "invalid") && !strings.Contains(combined, "role") {
		t.Fatalf("expected validation issues: %s", combined)
	}
}

func TestDoctorStrictMissingLlama(t *testing.T) {
	empty := t.TempDir()
	t.Setenv("PATH", empty)
	t.Setenv("QUIKAITOOLS_LLAMA", filepath.Join(empty, "no-such-llama"))
	var out, errb bytes.Buffer
	code := Main([]string{"quikaitools", "doctor", "--strict", "--profile", "cpu"}, &out, &errb)
	if code != 1 {
		t.Fatalf("strict cpu code=%d out=%s err=%s", code, out.String(), errb.String())
	}
}

func TestEmbedAllowStubFalse(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.onnx"), []byte("onnx"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := Main([]string{"quikaitools", "embed", "--model", dir, "--text", "hello", "--allow-stub=false"}, &out, &errb)
	if code != 1 {
		t.Fatalf("allow-stub=false code=%d out=%s err=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "stub") {
		t.Fatalf("expected stub in stderr: %s", errb.String())
	}
}
