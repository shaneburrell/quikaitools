package train

import (
	"fmt"
	"math"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/testart"
)

func TestRunLoRARandomLossFinite(t *testing.T) {
	losses, m, err := RunLoRARandom("the cat sat on the mat the cat sat", 4)
	if err != nil {
		t.Fatal(err)
	}
	if m.TrainableParams() == 0 {
		t.Fatal("no lora params")
	}
	line := ""
	for i, l := range losses {
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			t.Fatalf("step %d loss=%v", i, l)
		}
		if i > 0 {
			line += " "
		}
		line += fmt.Sprintf("%.4f", l)
	}
	testart.WriteFile(t, "lora-random-loss.txt", []byte(line+"\n"))
}
