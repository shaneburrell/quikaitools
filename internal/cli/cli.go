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
	"github.com/shaneburrell/quikaitools/internal/chatfmt"
	"github.com/shaneburrell/quikaitools/internal/exportx"
	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/hub"
	"github.com/shaneburrell/quikaitools/internal/infer"
	"github.com/shaneburrell/quikaitools/internal/peft"
	"github.com/shaneburrell/quikaitools/internal/train"
	"github.com/shaneburrell/quikaitools/internal/zoo"
)

const Version = "0.4.0"

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
	case "generate":
		return cmdGenerate(args[2:], stdout, stderr)
	case "embed":
		return cmdEmbed(args[2:], stdout, stderr)
	case "transcribe":
		return cmdTranscribe(args[2:], stdout, stderr)
	case "export":
		return cmdExport(args[2:], stdout, stderr)
	case "validate-sft":
		return cmdValidateSFT(args[2:], stdout, stderr)
	default:
		fmt.Fprintf(stderr, "unknown command %q\n\n", args[1])
		printUsage(stderr)
		return 2
	}
}

func printUsage(w io.Writer) {
	fmt.Fprint(w, `quikaitools — compose Go ML tools on V100, Strix Halo, and Mac

Usage:
  quikaitools doctor [--profile auto|v100|cuda|halo|mac|cpu] [--strict]
  quikaitools catalog [--task TASK] [--machine v100|halo|mac] [--catalog DIR]
  quikaitools pull [HF_REPO|catalog-id] [--cache DIR]
  quikaitools train lora|qlora --model DIR --data FILE [options]
  quikaitools generate --model DIR [--adapter DIR] [--prompt TEXT] [--tokens N]
      [--template chatml|raw] [--messages JSON]
  quikaitools generate --gguf FILE [--prompt TEXT] [--tokens N] [--profile KIND]
  quikaitools embed --model DIR --text TEXT
  quikaitools embed --vision --model DIR --image FILE
  quikaitools transcribe --model DIR --audio FILE
  quikaitools export merge|gguf|modelfile [flags]
  quikaitools validate-sft --path FILE.jsonl
  quikaitools version

Docs: https://github.com/shaneburrell/quikaitools
`)
}

func cmdDoctor(args []string, stdout, stderr io.Writer) int {
	profile := backend.KindAuto
	strict := false
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--profile":
			i++
			if i >= len(args) {
				fmt.Fprintln(stderr, "--profile needs a value")
				return 2
			}
			profile = backend.Kind(args[i])
		case "--strict":
			strict = true
		case "-h", "--help":
			fmt.Fprintln(stdout, "quikaitools doctor [--profile auto|v100|cuda|halo|mac|cpu] [--strict]")
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
	fmt.Fprintln(stdout, "binding:          Layer A labels — train is portable Go; ONNX needs Hugot/ORT; GGUF needs llama-cli")
	if p.MixedPrecision == "fp16" {
		fmt.Fprintln(stdout, "warning:          mixed_precision=fp16 is reserved; Go trainer compute is still FP32")
	}
	if p.Kind == backend.KindCPU {
		fmt.Fprintln(stdout, "\nwarning: landed on CPU. If this is a V100/Halo/Mac box, install the GPU SDK and re-run.")
	}
	if strict {
		ok, lines := infer.StrictCheck(p)
		for _, line := range lines {
			fmt.Fprintln(stdout, line)
		}
		if !ok {
			return 1
		}
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
			fmt.Fprintf(stdout, "quikaitools pull [HF_REPO|catalog-id] [--cache DIR]\nDefault: %s\n", TinyRepo)
			return 0
		default:
			if strings.HasPrefix(args[i], "-") {
				fmt.Fprintf(stderr, "unknown flag %s\n", args[i])
				return 2
			}
			repo = args[i]
		}
	}
	files := hub.DefaultFiles
	label := repo
	if cat, err := loadCatalog(""); err == nil {
		if m, ok := cat.Get(repo); ok {
			src := zoo.RepoFromSource(m.Source)
			if src != "" {
				fmt.Fprintf(stdout, "catalog %s → %s\n", m.ID, src)
				repo = src
				if fl := zoo.FilesForModel(m); len(fl) > 0 {
					files = fl
				}
				label = m.ID
			}
		}
	}
	c := hub.New(cache)
	dir, err := c.Pull(repo, files)
	if err != nil {
		// ONNX optional files often 404 — try GPT-2 defaults if safetensors pull partially failed
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintf(stdout, "pulled %s\n%s\n", label, dir)
	return 0
}

func cmdTrain(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" {
		fmt.Fprintln(stdout, "quikaitools train lora|qlora --model DIR --data FILE [--steps N] [--rank R] [--lr F] [--out DIR] [--accum N] [--resume DIR] [--profile KIND] [--eval-every N] [--smoke]")
		return 0
	}
	recipe := args[0]
	if recipe != "lora" && recipe != "qlora" {
		fmt.Fprintf(stderr, "unknown train recipe %q (want lora|qlora)\n", recipe)
		return 2
	}
	opt := train.LoRAOptions{Steps: 30, SeqLen: 32, Rank: 4, Alpha: 8, LR: 3e-3, Accum: 1, CkptEvery: 10, QLoRA: recipe == "qlora"}
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
		case "--resume":
			v, ok := need()
			if !ok {
				return 2
			}
			opt.Resume = v
		case "--profile":
			v, ok := need()
			if !ok {
				return 2
			}
			opt.Profile = backend.Kind(v)
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
		case "--accum":
			v, ok := need()
			if !ok {
				return 2
			}
			n, err := strconv.Atoi(v)
			if err != nil {
				fmt.Fprintln(stderr, err)
				return 2
			}
			opt.Accum = n
		case "--eval-every":
			v, ok := need()
			if !ok {
				return 2
			}
			n, err := strconv.Atoi(v)
			if err != nil {
				fmt.Fprintln(stderr, err)
				return 2
			}
			opt.EvalEvery = n
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
		case "--smoke":
			if opt.Steps > 30 {
				opt.Steps = 20
			}
			opt.Steps = 20
			opt.SeqLen = 32
			opt.Accum = 4
		default:
			fmt.Fprintf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if opt.ModelDir == "" || opt.DataPath == "" {
		fmt.Fprintln(stderr, "train requires --model and --data")
		return 2
	}
	if opt.OutDir == "" {
		suffix := "adapter-lora"
		if opt.QLoRA {
			suffix = "adapter-qlora"
		}
		opt.OutDir = filepath.Join(opt.ModelDir, suffix)
	}
	losses, err := train.RunLoRA(opt)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintf(stdout, "%s steps=%d rank=%d accum=%d done\n", recipe, len(losses), opt.Rank, opt.Accum)
	if len(losses) > 0 {
		fmt.Fprintf(stdout, "loss_first=%.4f loss_last=%.4f\n", losses[0], losses[len(losses)-1])
	}
	fmt.Fprintf(stdout, "adapter %s\n", opt.OutDir)
	return 0
}

func cmdGenerate(args []string, stdout, stderr io.Writer) int {
	var model, adapter, gguf, prompt, template, messagesJSON string
	tokens := 16
	profile := backend.KindAuto
	for i := 0; i < len(args); i++ {
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
			model = v
		case "--adapter":
			v, ok := need()
			if !ok {
				return 2
			}
			adapter = v
		case "--gguf":
			v, ok := need()
			if !ok {
				return 2
			}
			gguf = v
		case "--prompt":
			v, ok := need()
			if !ok {
				return 2
			}
			prompt = v
		case "--template":
			v, ok := need()
			if !ok {
				return 2
			}
			template = v
		case "--messages":
			v, ok := need()
			if !ok {
				return 2
			}
			messagesJSON = v
		case "--tokens":
			v, ok := need()
			if !ok {
				return 2
			}
			n, err := strconv.Atoi(v)
			if err != nil {
				fmt.Fprintln(stderr, err)
				return 2
			}
			tokens = n
		case "--profile":
			v, ok := need()
			if !ok {
				return 2
			}
			profile = backend.Kind(v)
		case "-h", "--help":
			fmt.Fprintln(stdout, "quikaitools generate --model DIR [--adapter DIR] | --gguf FILE [--prompt TEXT] [--template chatml|raw] [--messages JSON] [--tokens N]")
			return 0
		default:
			fmt.Fprintf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if messagesJSON != "" || template != "" {
		var msgs []chatfmt.Message
		if messagesJSON != "" {
			var err error
			msgs, err = chatfmt.ParseMessagesJSON(messagesJSON)
			if err != nil {
				fmt.Fprintln(stderr, err)
				return 2
			}
		}
		formatted, err := chatfmt.Apply(template, msgs, prompt)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
		prompt = formatted
	} else if prompt == "" {
		prompt = "Hello"
	}
	if gguf != "" {
		p, err := backend.Detect(profile, nil)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		out, err := infer.GenerateGGUF(infer.GenerateGGUFOptions{Model: gguf, Prompt: prompt, Tokens: tokens, Profile: p})
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		fmt.Fprintln(stdout, out)
		return 0
	}
	if model == "" {
		fmt.Fprintln(stderr, "generate requires --model or --gguf")
		return 2
	}
	base, err := gpt2.LoadDir(model)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	tok, err := gpt2.LoadTokenizer(model)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	var m *peft.Model
	if adapter != "" {
		m, err = peft.Load(base, adapter)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	} else {
		m = peft.BaseOnly(base)
	}
	ids := tok.Encode(prompt)
	if len(ids) == 0 {
		ids = []int{0}
	}
	out := m.Generate(ids, tokens)
	fmt.Fprintln(stdout, tok.Decode(out))
	return 0
}

func cmdEmbed(args []string, stdout, stderr io.Writer) int {
	visionMode := false
	var model, text, image string
	for i := 0; i < len(args); i++ {
		need := func() (string, bool) {
			i++
			if i >= len(args) {
				fmt.Fprintf(stderr, "%s needs a value\n", args[i-1])
				return "", false
			}
			return args[i], true
		}
		switch args[i] {
		case "--vision":
			visionMode = true
		case "--model":
			v, ok := need()
			if !ok {
				return 2
			}
			model = v
		case "--text":
			v, ok := need()
			if !ok {
				return 2
			}
			text = v
		case "--image":
			v, ok := need()
			if !ok {
				return 2
			}
			image = v
		case "-h", "--help":
			fmt.Fprintln(stdout, "quikaitools embed --model DIR --text TEXT | --vision --model DIR --image FILE")
			return 0
		default:
			fmt.Fprintf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if model == "" {
		fmt.Fprintln(stderr, "embed requires --model")
		return 2
	}
	if visionMode {
		if image == "" {
			fmt.Fprintln(stderr, "embed --vision requires --image")
			return 2
		}
		v, eng, err := infer.EmbedVision(infer.EmbedVisionOptions{ModelDir: model, Image: image, Size: 224})
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		fmt.Fprintf(stdout, "engine=%s dim=%d\n", eng, len(v))
		fmt.Fprintln(stdout, formatVec(v, 8))
		return 0
	}
	if text == "" {
		fmt.Fprintln(stderr, "embed requires --text (or --vision --image)")
		return 2
	}
	v, eng, err := infer.EmbedText(infer.EmbedTextOptions{ModelDir: model, Text: text})
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintf(stdout, "engine=%s dim=%d\n", eng, len(v))
	fmt.Fprintln(stdout, formatVec(v, 8))
	return 0
}

func cmdTranscribe(args []string, stdout, stderr io.Writer) int {
	var model, audioPath string
	for i := 0; i < len(args); i++ {
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
			model = v
		case "--audio":
			v, ok := need()
			if !ok {
				return 2
			}
			audioPath = v
		case "-h", "--help":
			fmt.Fprintln(stdout, "quikaitools transcribe --model DIR --audio FILE")
			return 0
		default:
			fmt.Fprintf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if model == "" || audioPath == "" {
		fmt.Fprintln(stderr, "transcribe requires --model and --audio")
		return 2
	}
	text, eng, err := infer.Transcribe(infer.TranscribeOptions{ModelDir: model, Audio: audioPath})
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintf(stdout, "engine=%s\n%s\n", eng, text)
	return 0
}

func formatVec(v []float32, n int) string {
	if n > len(v) {
		n = len(v)
	}
	parts := make([]string, n)
	for i := 0; i < n; i++ {
		parts[i] = fmt.Sprintf("%.4f", v[i])
	}
	return strings.Join(parts, " ")
}

func cmdValidateSFT(args []string, stdout, stderr io.Writer) int {
	path := ""
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--path":
			i++
			if i >= len(args) {
				fmt.Fprintln(stderr, "--path needs a value")
				return 2
			}
			path = args[i]
		case "-h", "--help":
			fmt.Fprintln(stdout, "quikaitools validate-sft --path FILE.jsonl")
			return 0
		default:
			fmt.Fprintf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if path == "" {
		fmt.Fprintln(stderr, "validate-sft requires --path")
		return 2
	}
	total, issues, err := train.ValidateMessagesJSONL(path)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if len(issues) == 0 {
		fmt.Fprintf(stdout, "VALID %s: %d rows\n", path, total)
		return 0
	}
	fmt.Fprintf(stdout, "INVALID %s: %d rows, %d issues\n", path, total, len(issues))
	for i, issue := range issues {
		if i >= 20 {
			break
		}
		fmt.Fprintf(stdout, "  - %s\n", issue)
	}
	return 1
}

func cmdExport(args []string, stdout, stderr io.Writer) int {
	if len(args) < 1 {
		fmt.Fprintln(stderr, "export requires merge|gguf|modelfile")
		return 2
	}
	sub := args[0]
	args = args[1:]
	switch sub {
	case "merge":
		return cmdExportMerge(args, stdout, stderr)
	case "gguf":
		return cmdExportGGUF(args, stdout, stderr)
	case "modelfile":
		return cmdExportModelfile(args, stdout, stderr)
	case "-h", "--help":
		fmt.Fprintln(stdout, "quikaitools export merge|gguf|modelfile …")
		return 0
	default:
		fmt.Fprintf(stderr, "unknown export subcommand %q\n", sub)
		return 2
	}
}

func cmdExportMerge(args []string, stdout, stderr io.Writer) int {
	model, adapter, out := "", "", ""
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--model":
			i++
			model = args[i]
		case "--adapter":
			i++
			adapter = args[i]
		case "--out":
			i++
			out = args[i]
		case "-h", "--help":
			fmt.Fprintln(stdout, "quikaitools export merge --model DIR --adapter DIR --out DIR")
			return 0
		default:
			fmt.Fprintf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if model == "" || adapter == "" || out == "" {
		fmt.Fprintln(stderr, "export merge requires --model, --adapter, and --out")
		return 2
	}
	if err := exportx.MergeGPT2LoRA(model, adapter, out); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintf(stdout, "merged -> %s\n", out)
	return 0
}

func cmdExportGGUF(args []string, stdout, stderr io.Writer) int {
	model, out, quant := "", "", "Q4_K_M"
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--model":
			i++
			model = args[i]
		case "--out":
			i++
			out = args[i]
		case "--quant":
			i++
			quant = args[i]
		case "-h", "--help":
			fmt.Fprintln(stdout, "quikaitools export gguf --model DIR --out FILE [--quant Q4_K_M]")
			return 0
		default:
			fmt.Fprintf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if model == "" || out == "" {
		fmt.Fprintln(stderr, "export gguf requires --model and --out")
		return 2
	}
	skip, err := exportx.ConvertGGUF(model, out, quant)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if skip != "" {
		fmt.Fprintf(stdout, "SKIP: %s\n", skip)
		return 0
	}
	fmt.Fprintf(stdout, "gguf -> %s\n", out)
	return 0
}

func cmdExportModelfile(args []string, stdout, stderr io.Writer) int {
	template, gguf, out := "", "", ""
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--template":
			i++
			template = args[i]
		case "--gguf":
			i++
			gguf = args[i]
		case "--out":
			i++
			out = args[i]
		case "-h", "--help":
			fmt.Fprintln(stdout, "quikaitools export modelfile --gguf FILE --out FILE [--template FILE]")
			return 0
		default:
			fmt.Fprintf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if gguf == "" || out == "" {
		fmt.Fprintln(stderr, "export modelfile requires --gguf and --out")
		return 2
	}
	if err := exportx.WriteModelfile(template, gguf, out); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintf(stdout, "modelfile -> %s\n", out)
	return 0
}
