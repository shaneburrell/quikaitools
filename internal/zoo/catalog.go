// Package zoo loads the machine-readable model catalog.
package zoo

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// Status is per-machine readiness in the catalog.
type Status string

const (
	StatusWorks    Status = "works"
	StatusCPUOnly  Status = "cpu_only"
	StatusUntested Status = "untested"
	StatusWont     Status = "wont"
)

// Model is one catalog entry (one YAML file).
type Model struct {
	ID       string            `yaml:"id"`
	Name     string            `yaml:"name"`
	Task     string            `yaml:"task"`
	Formats  []string          `yaml:"formats"`
	License  string            `yaml:"license"`
	Source   string            `yaml:"source"`
	Notes    string            `yaml:"notes"`
	Machines map[string]Status `yaml:"machines"`
	Engine   map[string]string `yaml:"engine"`
}

// Catalog is the loaded set.
type Catalog struct {
	Models []Model
}

// LoadDir reads every *.yaml file in dir (non-recursive except models/).
func LoadDir(dir string) (Catalog, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return Catalog{}, fmt.Errorf("catalog dir: %w", err)
	}
	var files []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, ".yaml") || strings.HasSuffix(name, ".yml") {
			files = append(files, filepath.Join(dir, name))
		}
	}
	if len(files) == 0 {
		// Allow catalog/models/
		nested := filepath.Join(dir, "models")
		if info, err := os.Stat(nested); err == nil && info.IsDir() {
			return LoadDir(nested)
		}
	}
	sort.Strings(files)
	var cat Catalog
	for _, path := range files {
		m, err := loadFile(path)
		if err != nil {
			return Catalog{}, err
		}
		cat.Models = append(cat.Models, m)
	}
	return cat, nil
}

// LoadFS loads YAML models from an fs.FS rooted at the models directory.
func LoadFS(fsys fs.FS) (Catalog, error) {
	var files []string
	err := fs.WalkDir(fsys, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if strings.HasSuffix(path, ".yaml") || strings.HasSuffix(path, ".yml") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return Catalog{}, err
	}
	sort.Strings(files)
	var cat Catalog
	for _, path := range files {
		b, err := fs.ReadFile(fsys, path)
		if err != nil {
			return Catalog{}, err
		}
		m, err := parseModel(b, path)
		if err != nil {
			return Catalog{}, err
		}
		cat.Models = append(cat.Models, m)
	}
	return cat, nil
}

func loadFile(path string) (Model, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Model{}, err
	}
	return parseModel(b, path)
}

func parseModel(b []byte, path string) (Model, error) {
	var m Model
	if err := yaml.Unmarshal(b, &m); err != nil {
		return Model{}, fmt.Errorf("%s: %w", path, err)
	}
	if m.ID == "" {
		return Model{}, fmt.Errorf("%s: missing id", path)
	}
	if m.Task == "" {
		return Model{}, fmt.Errorf("%s: missing task", path)
	}
	return m, nil
}

// Filter returns models matching optional task and machine status.
// machine is a catalog column (v100, halo, mac, cpu). Empty means all.
func (c Catalog) Filter(task, machine string) []Model {
	task = strings.TrimSpace(strings.ToLower(task))
	machine = strings.TrimSpace(strings.ToLower(machine))
	var out []Model
	for _, m := range c.Models {
		if task != "" && strings.ToLower(m.Task) != task {
			continue
		}
		if machine != "" && machine != "cpu" {
			st, ok := m.Machines[machine]
			if !ok || st == StatusWont {
				continue
			}
		}
		out = append(out, m)
	}
	return out
}

// StatusOn returns the machine status or untested.
func (m Model) StatusOn(machine string) Status {
	if m.Machines == nil {
		return StatusUntested
	}
	if st, ok := m.Machines[machine]; ok {
		return st
	}
	return StatusUntested
}

// EngineOn returns the recommended engine for a machine.
func (m Model) EngineOn(machine string) string {
	if m.Engine == nil {
		return ""
	}
	return m.Engine[machine]
}

// Get returns the model with id, or false.
func (c Catalog) Get(id string) (Model, bool) {
	id = strings.TrimSpace(id)
	for _, m := range c.Models {
		if m.ID == id {
			return m, true
		}
	}
	return Model{}, false
}

// RepoFromSource extracts org/name from a Hugging Face URL or bare repo id.
func RepoFromSource(source string) string {
	source = strings.TrimSpace(source)
	if source == "" {
		return ""
	}
	const prefix = "https://huggingface.co/"
	if strings.HasPrefix(source, prefix) {
		rest := strings.TrimPrefix(source, prefix)
		parts := strings.Split(rest, "/")
		if len(parts) >= 2 {
			return parts[0] + "/" + parts[1]
		}
	}
	if strings.Count(source, "/") == 1 && !strings.Contains(source, "://") {
		return source
	}
	return source
}

// FilesForModel picks Hub files to download based on formats.
func FilesForModel(m Model) []string {
	fmts := map[string]bool{}
	for _, f := range m.Formats {
		fmts[strings.ToLower(f)] = true
	}
	var files []string
	seen := map[string]bool{}
	add := func(name string) {
		if !seen[name] {
			seen[name] = true
			files = append(files, name)
		}
	}
	if fmts["safetensors"] {
		for _, f := range []string{"config.json", "model.safetensors", "vocab.json", "merges.txt", "tokenizer_config.json"} {
			add(f)
		}
	}
	if fmts["onnx"] {
		add("config.json")
		add("tokenizer.json")
		add("tokenizer_config.json")
		add("vocab.txt")
		add("onnx/model.onnx")
		add("model.onnx")
	}
	if fmts["gguf"] {
		add("README.md")
	}
	if len(files) == 0 {
		return nil // hub default
	}
	return files
}
