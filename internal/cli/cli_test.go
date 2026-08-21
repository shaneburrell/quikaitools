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
	if !strings.Contains(out.String(), "doctor") {
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
