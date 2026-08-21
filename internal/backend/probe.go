package backend

import (
	"os"
	"os/exec"
	"regexp"
	"strings"
)

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func runLookup(name string, args ...string) string {
	if _, err := exec.LookPath(name); err != nil {
		return ""
	}
	out, err := exec.Command(name, args...).CombinedOutput()
	if err != nil {
		return ""
	}
	return string(out)
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[:i])
	}
	return s
}

var gfxRe = regexp.MustCompile(`gfx[0-9a-f]+`)

func detectGFX(rocminfo string) string {
	return gfxRe.FindString(strings.ToLower(rocminfo))
}
