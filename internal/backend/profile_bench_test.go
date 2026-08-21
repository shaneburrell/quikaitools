package backend

import "testing"

func BenchmarkInferKind(b *testing.B) {
	hosts := []HostInfo{
		{GOOS: "darwin"},
		{GOOS: "linux", NvidiaName: "Tesla V100-SXM2-32GB"},
		{GOOS: "linux", HasKFD: true, ROCmGFX: "gfx1151"},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = inferKind(hosts[i%len(hosts)])
	}
}
