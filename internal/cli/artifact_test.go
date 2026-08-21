package cli

import (
	"bytes"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/backend"
	"github.com/shaneburrell/quikaitools/internal/testart"
)

// Writes doctor and catalog dumps under testdata/artifacts (gitignored).
func TestWriteDoctorAndCatalogArtifacts(t *testing.T) {
	for _, kind := range []backend.Kind{
		backend.KindV100, backend.KindCUDA, backend.KindHalo, backend.KindMac, backend.KindCPU,
	} {
		var out, errb bytes.Buffer
		code := Main([]string{"quikaitools", "doctor", "--profile", string(kind)}, &out, &errb)
		if code != 0 {
			t.Fatalf("doctor %s: code=%d err=%s", kind, code, errb.String())
		}
		testart.WriteFile(t, "doctor-"+string(kind)+".txt", out.Bytes())
	}

	for _, machine := range []string{"v100", "halo", "mac"} {
		var out, errb bytes.Buffer
		code := Main([]string{"quikaitools", "catalog", "--machine", machine}, &out, &errb)
		if code != 0 {
			t.Fatalf("catalog %s: code=%d err=%s", machine, code, errb.String())
		}
		testart.WriteFile(t, "catalog-"+machine+".txt", out.Bytes())
	}
}
