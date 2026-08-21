package cli

import (
	"bytes"
	"image/color"
	"os"
	"path/filepath"
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
	if code := Main([]string{"quikaitools", "help"}, &out, &out); code != 0 {
		t.Fatalf("help code=%d", code)
	}
	if !strings.Contains(out.String(), "doctor") || !strings.Contains(out.String(), "train") {
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
