package backend

import "testing"

func TestDetectExplicit(t *testing.T) {
	p, err := Detect(KindV100, func() HostInfo { return HostInfo{GOOS: "linux"} })
	if err != nil {
		t.Fatal(err)
	}
	if p.Kind != KindV100 || p.MixedPrecision != "fp16" || p.TrainEngine != "gomlx-xla-cuda" {
		t.Fatalf("v100 profile = %+v", p)
	}
}

func TestInferKind(t *testing.T) {
	cases := []struct {
		name string
		host HostInfo
		want Kind
	}{
		{"mac", HostInfo{GOOS: "darwin"}, KindMac},
		{"v100", HostInfo{GOOS: "linux", NvidiaName: "Tesla V100-SXM2-32GB"}, KindV100},
		{"cuda", HostInfo{GOOS: "linux", NvidiaName: "NVIDIA RTX 4090"}, KindCUDA},
		{"halo kfd", HostInfo{GOOS: "linux", HasKFD: true}, KindHalo},
		{"halo gfx", HostInfo{GOOS: "linux", ROCmGFX: "gfx1151"}, KindHalo},
		{"cpu", HostInfo{GOOS: "linux"}, KindCPU},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := inferKind(tc.host)
			if got != tc.want {
				t.Fatalf("inferKind(%+v) = %s, want %s", tc.host, got, tc.want)
			}
		})
	}
}

func TestDetectUnknown(t *testing.T) {
	if _, err := Detect(Kind("nope"), func() HostInfo { return HostInfo{} }); err == nil {
		t.Fatal("expected error")
	}
}

func TestCatalogMachineCUDAMapsToV100(t *testing.T) {
	p, err := Detect(KindCUDA, func() HostInfo { return HostInfo{} })
	if err != nil {
		t.Fatal(err)
	}
	if p.CatalogMachine() != "v100" {
		t.Fatalf("catalog machine = %s", p.CatalogMachine())
	}
}

func TestDetectGFX(t *testing.T) {
	got := detectGFX("Name: gfx1151\nMarketing Name: Radeon 8060S")
	if got != "gfx1151" {
		t.Fatalf("got %q", got)
	}
}
