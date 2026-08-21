// Package exportx merges LoRA adapters and prepares GGUF/Ollama export helpers.
package exportx

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/peft"
)

// MergeGPT2LoRA merges LoRA adapters into base GPT-2 and writes a HF-like dir.
func MergeGPT2LoRA(modelDir, adapterDir, outDir string) error {
	base, err := gpt2.LoadDir(modelDir)
	if err != nil {
		return err
	}
	if err := peft.LoadAndMerge(base, adapterDir); err != nil {
		return err
	}
	if err := gpt2.WriteDir(outDir, base); err != nil {
		return err
	}
	for _, name := range []string{"tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"} {
		src := filepath.Join(modelDir, name)
		dst := filepath.Join(outDir, name)
		if b, err := os.ReadFile(src); err == nil {
			_ = os.WriteFile(dst, b, 0o644)
		}
	}
	return nil
}

// WriteModelfile writes an Ollama Modelfile pointing at ggufPath.
func WriteModelfile(templatePath, ggufPath, outPath string) error {
	b, err := os.ReadFile(templatePath)
	if err != nil {
		b = []byte("FROM {{GGUF}}\nPARAMETER temperature 0.2\n")
	}
	text := string(b)
	lines := strings.Split(text, "\n")
	replaced := false
	for i, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "FROM ") {
			lines[i] = "FROM " + ggufPath
			replaced = true
			break
		}
	}
	text = strings.Join(lines, "\n")
	text = strings.ReplaceAll(text, "{{GGUF}}", ggufPath)
	if !replaced && !strings.Contains(text, ggufPath) {
		text = "FROM " + ggufPath + "\n" + text
	}
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	return os.WriteFile(outPath, []byte(text), 0o644)
}

// ConvertGGUF looks for llama.cpp convert tools; returns skip reason if unavailable.
// QuikAITools does not vendor Python; skip is success when tools are missing.
func ConvertGGUF(modelDir, outGGUF, quant string) (skipped string, err error) {
	if quant == "" {
		quant = "Q4_K_M"
	}
	convert := findOnPath("convert_hf_to_gguf.py", "convert-hf-to-gguf.py")
	quantize := findOnPath("llama-quantize", "quantize")
	if convert == "" {
		return "no convert_hf_to_gguf.py on PATH — supply a GGUF manually or install llama.cpp convert tools", nil
	}
	if err := os.MkdirAll(filepath.Dir(outGGUF), 0o755); err != nil {
		return "", err
	}
	tmpF16 := outGGUF + ".f16.gguf"
	cmd := exec.Command("python3", convert, modelDir, "--outfile", tmpF16, "--outtype", "f16")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("convert: %w\n%s", err, string(out))
	}
	if quantize == "" {
		if err := os.Rename(tmpF16, outGGUF); err != nil {
			return "", err
		}
		return "quantizer missing — left F16 GGUF at " + outGGUF, nil
	}
	cmd = exec.Command(quantize, tmpF16, outGGUF, quant)
	out, err = cmd.CombinedOutput()
	_ = os.Remove(tmpF16)
	if err != nil {
		return "", fmt.Errorf("quantize: %w\n%s", err, string(out))
	}
	return "", nil
}

func findOnPath(names ...string) string {
	for _, n := range names {
		if p, err := exec.LookPath(n); err == nil {
			return p
		}
	}
	return ""
}
