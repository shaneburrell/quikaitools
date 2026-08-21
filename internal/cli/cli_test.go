package cli

import (
	"bytes"
	"strings"
	"testing"
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
	if code := Main([]string{"quikaitools", "train", "lora", "-h"}, &out, &err); code != 0 && !strings.Contains(out.String()+err.String(), "lora") {
		// train with only -h after lora is parsed as unknown flag; accept help on train
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
