package cli

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/shaneburrell/quikaitools/catalog"
	"github.com/shaneburrell/quikaitools/internal/backend"
	"github.com/shaneburrell/quikaitools/internal/hub"
	"github.com/shaneburrell/quikaitools/internal/train"
	"github.com/shaneburrell/quikaitools/internal/zoo"
)

const Version = "0.2.0"

// Main is the CLI entrypoint. args[0] is the program name.
func Main(args []string, stdout, stderr io.Writer) int {
	if len(args) < 2 || args[1] == "-h" || args[1] == "--help" || args[1] == "help" {
		printUsage(stdout)
		return 0
	}
	switch args[1] {
	case "version", "--version":
		fmt.Fprintln(stdout, Version)
		return 0
	case "doctor":
		return cmdDoctor(args[2:], stdout, stderr)
	case "catalog":
		return cmdCatalog(args[2:], stdout, stderr)
	case "pull":
		return cmdPull(args[2:], stdout, stderr)
	case "train":
		return cmdTrain(args[2:], stdout, stderr)
	default:
		fmt.Fprintf(stderr, "unknown command %q\n\n", args[1])
		printUsage(stderr)
		return 2
	}
}

func printUsage(w io.Writer) {
	fmt.Fprint(w, `quikaitools — compose Go ML tools on V100, Strix Halo, and Mac

Usage:
  quikaitools doctor [--profile auto|v100|cuda|halo|mac|cpu]
  quikaitools catalog [--task TASK] [--machine v100|halo|mac] [--catalog DIR]
  quikaitools pull [HF_REPO]
  quikaitools train lora --model DIR --data FILE [--steps N] [--rank R] [--out DIR]
  quikaitools version

Default pull repo: hf-internal-testing/tiny-random-gpt2 (32d, 5 layers, ~450KB).

Docs: https://github.com/shaneburrell/quikaitools
`)
}

func cmdDoctor(args []string, stdout, stderr io.Writer) int {
	profile := backend.KindAuto
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--profile":
			i++
			if i >= len(args) {
				fmt.Fprintln(stderr, "--profile needs a value")
				return 2
			}
			profile = backend.Kind(args[i])
		case "-h", "--help":
			fmt.Fprintln(stdout, "quikaitools doctor [--profile auto|v100|cuda|halo|mac|cpu]")
			return 0
		default:
			fmt.Fprintf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	p, err := backend.Detect(profile, nil)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintf(stdout, "profile:          %s\n", p)
	fmt.Fprintf(stdout, "detected_from:    %s\n", p.DetectedFrom)
	if p.HardwareHint != "" {
		fmt.Fprintf(stdout, "hardware:         %s\n", p.HardwareHint)
	}
	fmt.Fprintf(stdout, "train:            %s\n", p.TrainEngine)
	fmt.Fprintf(stdout, "infer_onnx:       %s\n", p.InferONNX)
	fmt.Fprintf(stdout, "infer_gguf:       %s\n", p.InferGGUF)
	fmt.Fprintf(stdout, "mixed_precision:  %s\n", p.MixedPrecision)
	fmt.Fprintf(stdout, "notes:            %s\n", p.Notes)
	if p.Kind == backend.KindCPU {
		fmt.Fprintln(stdout, "\nwarning: landed on CPU. If this is a V100/Halo/Mac box, install the GPU SDK and re-run.")
	}
	return 0
}

func cmdCatalog(args []string, stdout, stderr io.Writer) int {
	var task, machine, dir string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--task":
			i++
			if i >= len(args) {
				fmt.Fprintln(stderr, "--task needs a value")
				return 2
			}
			task = args[i]
		case "--machine":
			i++
			if i >= len(args) {
				fmt.Fprintln(stderr, "--machine needs a value")
				return 2
			}
			machine = args[i]
		case "--catalog":
			i++
			if i >= len(args) {
				fmt.Fprintln(stderr, "--catalog needs a value")
				return 2
			}
			dir = args[i]
		case "-h", "--help":
			fmt.Fprintln(stdout, "quikaitools catalog [--task TASK] [--machine v100|halo|mac] [--catalog DIR]")
			return 0
		default:
			fmt.Fprintf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}

	cat, err := loadCatalog(dir)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if machine == "" {
		if p, err := backend.Detect(backend.KindAuto, nil); err == nil {
			machine = p.CatalogMachine()
		}
	}
	models := cat.Filter(task, machine)
	if len(models) == 0 {
		fmt.Fprintln(stdout, "no models matched")
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "ID\tTASK\tFORMATS\tSTATUS\tENGINE")
	for _, m := range models {
		st := m.StatusOn(machine)
		eng := m.EngineOn(machine)
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", m.ID, m.Task, join(m.Formats), st, eng)
	}
	_ = tw.Flush()
	return 0
}

func loadCatalog(dir string) (zoo.Catalog, error) {
	if dir == "" {
		dir = os.Getenv("QUIKAITOOLS_CATALOG")
	}
	if dir == "" {
		if cwd, err := os.Getwd(); err == nil {
			cand := filepath.Join(cwd, "catalog")
			if st, err := os.Stat(cand); err == nil && st.IsDir() {
				dir = cand
			}
		}
	}
	if dir != "" {
		return zoo.LoadDir(dir)
	}
	return zoo.LoadFS(catalog.Models)
}

func join(s []string) string {
	return strings.Join(s, ",")
}

// TinyRepo is the default LoRA lab model (~450KB safetensors).
const TinyRepo = "hf-internal-testing/tiny-random-gpt2"

func cmdPull(args []string, stdout, stderr io.Writer) int {
	repo := TinyRepo
	cache := ""
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--cache":
			i++
			if i >= len(args) {
				fmt.Fprintln(stderr, "--cache needs a value")
				return 2
			}
			cache = args[i]
		case "-h", "--help":
			fmt.Fprintf(stdout, "quikaitools pull [HF_REPO] [--cache DIR]\nDefault repo: %s\n", TinyRepo)
			return 0
		default:
			if strings.HasPrefix(args[i], "-") {
				fmt.Fprintf(stderr, "unknown flag %s\n", args[i])
				return 2
			}
			repo = args[i]
		}
	}
	c := hub.New(cache)
	dir, err := c.Pull(repo, nil)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintf(stdout, "pulled %s\n%s\n", repo, dir)
	return 0
}

func cmdTrain(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" {
		fmt.Fprintln(stdout, "quikaitools train lora --model DIR --data FILE [--steps N] [--rank R] [--lr F] [--out DIR]")
		return 0
	}
	if args[0] != "lora" {
		fmt.Fprintf(stderr, "unknown train recipe %q (want lora)\n", args[0])
		return 2
	}
	opt := train.LoRAOptions{Steps: 30, SeqLen: 32, Rank: 4, Alpha: 8, LR: 3e-3}
	for i := 1; i < len(args); i++ {
		need := func() (string, bool) {
			i++
			if i >= len(args) {
				fmt.Fprintf(stderr, "%s needs a value\n", args[i-1])
				return "", false
			}
			return args[i], true
		}
		switch args[i] {
		case "--model":
			v, ok := need()
			if !ok {
				return 2
			}
			opt.ModelDir = v
		case "--data":
			v, ok := need()
			if !ok {
				return 2
			}
			opt.DataPath = v
		case "--out":
			v, ok := need()
			if !ok {
				return 2
			}
			opt.OutDir = v
		case "--steps":
			v, ok := need()
			if !ok {
				return 2
			}
			n, err := strconv.Atoi(v)
			if err != nil {
				fmt.Fprintln(stderr, err)
				return 2
			}
			opt.Steps = n
		case "--rank":
			v, ok := need()
			if !ok {
				return 2
			}
			n, err := strconv.Atoi(v)
			if err != nil {
				fmt.Fprintln(stderr, err)
				return 2
			}
			opt.Rank = n
		case "--lr":
			v, ok := need()
			if !ok {
				return 2
			}
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				fmt.Fprintln(stderr, err)
				return 2
			}
			opt.LR = f
		default:
			fmt.Fprintf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if opt.ModelDir == "" || opt.DataPath == "" {
		fmt.Fprintln(stderr, "train lora requires --model and --data")
		return 2
	}
	if opt.OutDir == "" {
		opt.OutDir = filepath.Join(opt.ModelDir, "adapter-lora")
	}
	losses, err := train.RunLoRA(opt)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintf(stdout, "lora steps=%d rank=%d trainable done\n", len(losses), opt.Rank)
	if len(losses) > 0 {
		fmt.Fprintf(stdout, "loss_first=%.4f loss_last=%.4f\n", losses[0], losses[len(losses)-1])
	}
	fmt.Fprintf(stdout, "adapter %s\n", opt.OutDir)
	return 0
}
