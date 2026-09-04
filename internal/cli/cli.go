package cli

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

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

const Version = "0.5.0"

func outf(w io.Writer, format string, a ...any) {
	_, _ = fmt.Fprintf(w, format, a...)
}

func outln(w io.Writer, a ...any) {
	_, _ = fmt.Fprintln(w, a...)
}

func nextArg(args []string, i *int, stderr io.Writer) (string, bool) {
	flag := args[*i]
	*i++
	if *i >= len(args) || strings.HasPrefix(args[*i], "--") {
		outf(stderr, "%s needs a value\n", flag)
		return "", false
	}
	return args[*i], true
}

func parsePosInt(v, flag string, stderr io.Writer) (int, bool) {
	n, err := strconv.Atoi(v)
	if err != nil {
		outln(stderr, err)
		return 0, false
	}
	if n <= 0 {
		outf(stderr, "%s must be > 0\n", flag)
		return 0, false
	}
	return n, true
}

func parsePosFloat(v, flag string, stderr io.Writer) (float64, bool) {
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		outln(stderr, err)
		return 0, false
	}
	if f <= 0 {
		outf(stderr, "%s must be > 0\n", flag)
		return 0, false
	}
	return f, true
}

func parseNonNegInt(v, flag string, stderr io.Writer) (int, bool) {
	n, err := strconv.Atoi(v)
	if err != nil {
		outln(stderr, err)
		return 0, false
	}
	if n < 0 {
		outf(stderr, "%s must be >= 0\n", flag)
		return 0, false
	}
	return n, true
}

func parseNonNegFloat(v, flag string, stderr io.Writer) (float64, bool) {
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		outln(stderr, err)
		return 0, false
	}
	if f < 0 {
		outf(stderr, "%s must be >= 0\n", flag)
		return 0, false
	}
	return f, true
}

func parseUnitInterval(v, flag string, stderr io.Writer) (float64, bool) {
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		outln(stderr, err)
		return 0, false
	}
	if f < 0 || f > 1 {
		outf(stderr, "%s must be between 0 and 1\n", flag)
		return 0, false
	}
	return f, true
}

func parseInt64Arg(v, flag string, stderr io.Writer) (int64, bool) {
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		outf(stderr, "%s: %v\n", flag, err)
		return 0, false
	}
	return n, true
}

// boolFromFlag parses --flag (true), --flag=true, and --flag=false.
func boolFromFlag(arg string, stderr io.Writer) (bool, bool) {
	if i := strings.IndexByte(arg, '='); i >= 0 {
		name, raw := arg[:i], arg[i+1:]
		switch strings.ToLower(raw) {
		case "true", "1":
			return true, true
		case "false", "0":
			return false, true
		default:
			outf(stderr, "%s wants true or false\n", name)
			return false, false
		}
	}
	return true, true
}

func noteStubEngine(stderr io.Writer, engine string) {
	if strings.Contains(engine, "stub") {
		outf(stderr, "note: %s is a stub; ONNX runtime not linked\n", engine)
	}
}

func doctorUsage() string {
	kinds := make([]string, 0, 1+len(backend.AllKinds))
	kinds = append(kinds, string(backend.KindAuto))
	for _, k := range backend.AllKinds {
		kinds = append(kinds, string(k))
	}
	return "quikaitools doctor [--profile " + strings.Join(kinds, "|") + "] [--strict]"
}

// Main is the CLI entrypoint. args[0] is the program name.
func Main(args []string, stdout, stderr io.Writer) int {
	if len(args) < 2 || args[1] == "-h" || args[1] == "--help" || args[1] == "help" {
		printUsage(stdout)
		return 0
	}
	switch args[1] {
	case "version", "--version":
		outln(stdout, Version)
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
		outf(stderr, "unknown command %q\n\n", args[1])
		printUsage(stderr)
		return 2
	}
}

func printUsage(w io.Writer) {
	outf(w, `quikaitools — compose Go ML tools on V100, Strix Halo, and Mac

Usage:
  %s
  quikaitools catalog [--task TASK] [--machine v100|halo|mac|cpu] [--catalog DIR]
  quikaitools pull [HF_REPO|catalog-id] [--cache DIR]
  quikaitools train lora|qlora --model DIR --data FILE [--steps N] [--rank R] [--lr F]
      [--out DIR] [--accum N] [--resume DIR] [--profile KIND] [--eval-every N]
      [--smoke] [--seq N] [--seed N] [--mask-prompt]
  quikaitools generate --model DIR [--adapter DIR] [--prompt TEXT] [--tokens N]
      [--template chatml|raw] [--messages JSON]
      [--temperature F] [--top-k N] [--top-p F] [--seed N] [--stop-eos]
  quikaitools generate --gguf FILE [--prompt TEXT] [--tokens N] [--profile KIND] [--timeout DURATION]
  quikaitools embed --model DIR --text TEXT [--allow-stub]
  quikaitools embed --vision --model DIR --image FILE [--normalize imagenet|clip] [--allow-stub]
  quikaitools transcribe --model DIR --audio FILE [--allow-stub]
  quikaitools export merge|gguf|modelfile [flags]
  quikaitools validate-sft --path FILE.jsonl
  quikaitools version

Docs: https://github.com/shaneburrell/quikaitools
`, doctorUsage())
}

func cmdDoctor(args []string, stdout, stderr io.Writer) int {
	profile := backend.KindAuto
	strict := false
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--profile":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			profile = backend.Kind(v)
		case "--strict":
			strict = true
		case "-h", "--help":
			outln(stdout, doctorUsage())
			return 0
		default:
			outf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	p, err := backend.Detect(profile, nil)
	if err != nil {
		outln(stderr, err)
		return 1
	}
	outf(stdout, "profile:          %s\n", p)
	outf(stdout, "detected_from:    %s\n", p.DetectedFrom)
	if p.HardwareHint != "" {
		outf(stdout, "hardware:         %s\n", p.HardwareHint)
	}
	outf(stdout, "train:            %s\n", p.TrainEngine)
	outf(stdout, "infer_onnx:       %s\n", p.InferONNX)
	outf(stdout, "infer_gguf:       %s\n", p.InferGGUF)
	outf(stdout, "mixed_precision:  %s\n", p.MixedPrecision)
	outf(stdout, "notes:            %s\n", p.Notes)
	outln(stdout, "binding:          Layer A labels — train is portable Go; ONNX needs Hugot/ORT; GGUF needs llama-cli")
	if p.MixedPrecision == "fp16" {
		outln(stdout, "warning:          mixed_precision=fp16 is reserved; Go trainer compute is still FP32")
	}
	if p.Kind == backend.KindCPU {
		outln(stdout, "\nwarning: landed on CPU. If this is a V100/Halo/Mac box, install the GPU SDK and re-run.")
	}
	if strict {
		ok, lines := infer.StrictCheck(p)
		for _, line := range lines {
			outln(stdout, line)
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
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			task = v
		case "--machine":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			machine = v
		case "--catalog":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			dir = v
		case "-h", "--help":
			outln(stdout, "quikaitools catalog [--task TASK] [--machine v100|halo|mac|cpu] [--catalog DIR]")
			return 0
		default:
			outf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}

	cat, err := loadCatalog(dir)
	if err != nil {
		outln(stderr, err)
		return 1
	}
	if machine == "" {
		if p, err := backend.Detect(backend.KindAuto, nil); err == nil {
			machine = p.CatalogMachine()
		} else {
			outf(stderr, "warning: %s\n", err)
		}
	}
	models := cat.Filter(task, machine)
	if len(models) == 0 {
		outln(stdout, "no models matched")
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	outln(tw, "ID\tTASK\tFORMATS\tSTATUS\tENGINE")
	for _, m := range models {
		st := m.StatusOn(machine)
		eng := m.EngineOn(machine)
		outf(tw, "%s\t%s\t%s\t%s\t%s\n", m.ID, m.Task, join(m.Formats), st, eng)
	}
	if err := tw.Flush(); err != nil {
		outln(stderr, err)
		return 1
	}
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

// tinyRepo is the default LoRA lab model (~450KB safetensors).
const tinyRepo = "hf-internal-testing/tiny-random-gpt2"

func cmdPull(args []string, stdout, stderr io.Writer) int {
	repo := tinyRepo
	cache := ""
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--cache":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			cache = v
		case "-h", "--help":
			outf(stdout, "quikaitools pull [HF_REPO|catalog-id] [--cache DIR]\nDefault: %s\n", tinyRepo)
			return 0
		default:
			if strings.HasPrefix(args[i], "-") {
				outf(stderr, "unknown flag %s\n", args[i])
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
				outf(stdout, "catalog %s → %s\n", m.ID, src)
				repo = src
				if fl := zoo.FilesForModel(m); len(fl) > 0 {
					files = fl
				}
				label = m.ID
			}
		}
	} else {
		outf(stderr, "warning: %s\n", err)
	}
	c := hub.New(cache)
	dir, err := c.Pull(repo, files)
	if err != nil {
		outln(stderr, err)
		return 1
	}
	outf(stdout, "pulled %s\n%s\n", label, dir)
	return 0
}

func cmdTrain(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" {
		outln(stdout, "quikaitools train lora|qlora --model DIR --data FILE [--steps N] [--rank R] [--lr F] [--out DIR] [--accum N] [--resume DIR] [--profile KIND] [--eval-every N] [--smoke] [--seq N] [--seed N] [--mask-prompt]")
		return 0
	}
	recipe := args[0]
	if recipe != "lora" && recipe != "qlora" {
		outf(stderr, "unknown train recipe %q (want lora|qlora)\n", recipe)
		return 2
	}
	opt := train.LoRAOptions{Steps: 30, SeqLen: 32, Rank: 4, Alpha: 8, LR: 3e-3, Accum: 1, CkptEvery: 10, QLoRA: recipe == "qlora"}
	smoke := false
	for i := 1; i < len(args); i++ {
		switch args[i] {
		case "--model":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			opt.ModelDir = v
		case "--data":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			opt.DataPath = v
		case "--out":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			opt.OutDir = v
		case "--resume":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			opt.Resume = v
		case "--profile":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			opt.Profile = backend.Kind(v)
		case "--steps":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			n, ok := parsePosInt(v, "--steps", stderr)
			if !ok {
				return 2
			}
			opt.Steps = n
		case "--rank":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			n, ok := parsePosInt(v, "--rank", stderr)
			if !ok {
				return 2
			}
			opt.Rank = n
		case "--accum":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			n, ok := parsePosInt(v, "--accum", stderr)
			if !ok {
				return 2
			}
			opt.Accum = n
		case "--eval-every":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			n, ok := parsePosInt(v, "--eval-every", stderr)
			if !ok {
				return 2
			}
			opt.EvalEvery = n
		case "--lr":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			f, ok := parsePosFloat(v, "--lr", stderr)
			if !ok {
				return 2
			}
			opt.LR = f
		case "--smoke":
			smoke = true
		case "--seq":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			n, ok := parsePosInt(v, "--seq", stderr)
			if !ok {
				return 2
			}
			opt.SeqLen = n
		case "--seed":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			n, ok := parseInt64Arg(v, "--seed", stderr)
			if !ok {
				return 2
			}
			opt.Seed = n
		case "--mask-prompt", "--mask-prompt=true", "--mask-prompt=false":
			b, ok := boolFromFlag(args[i], stderr)
			if !ok {
				return 2
			}
			opt.MaskPrompt = b
		default:
			outf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if smoke {
		opt.Steps = 20
		opt.SeqLen = 32
		opt.Accum = 4
	}
	if opt.ModelDir == "" || opt.DataPath == "" {
		outln(stderr, "train requires --model and --data")
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
		outln(stderr, err)
		return 1
	}
	outf(stdout, "%s steps=%d optimizer_steps=%d rank=%d accum=%d done\n", recipe, opt.Steps, len(losses), opt.Rank, opt.Accum)
	if len(losses) > 0 {
		outf(stdout, "loss_first=%.4f loss_last=%.4f\n", losses[0], losses[len(losses)-1])
	}
	outf(stdout, "adapter %s\n", opt.OutDir)
	return 0
}

func cmdGenerate(args []string, stdout, stderr io.Writer) int {
	var model, adapter, gguf, prompt, template, messagesJSON string
	tokens := 16
	profile := backend.KindAuto
	var temperature float64
	var topK int
	var topP float64
	var seed int64
	var timeout time.Duration
	var stopEOS, tempSet, topKSet, topPSet, seedSet bool
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--model":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			model = v
		case "--adapter":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			adapter = v
		case "--gguf":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			gguf = v
		case "--prompt":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			prompt = v
		case "--template":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			template = v
		case "--messages":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			messagesJSON = v
		case "--tokens":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			n, ok := parsePosInt(v, "--tokens", stderr)
			if !ok {
				return 2
			}
			tokens = n
		case "--profile":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			profile = backend.Kind(v)
		case "--temperature":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			f, ok := parseNonNegFloat(v, "--temperature", stderr)
			if !ok {
				return 2
			}
			temperature = f
			tempSet = true
		case "--top-k":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			n, ok := parseNonNegInt(v, "--top-k", stderr)
			if !ok {
				return 2
			}
			topK = n
			topKSet = true
		case "--top-p":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			f, ok := parseUnitInterval(v, "--top-p", stderr)
			if !ok {
				return 2
			}
			topP = f
			topPSet = true
		case "--seed":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			n, ok := parseInt64Arg(v, "--seed", stderr)
			if !ok {
				return 2
			}
			seed = n
			seedSet = true
		case "--stop-eos", "--stop-eos=true", "--stop-eos=false":
			b, ok := boolFromFlag(args[i], stderr)
			if !ok {
				return 2
			}
			stopEOS = b
		case "--timeout":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			d, err := time.ParseDuration(v)
			if err != nil {
				outln(stderr, err)
				return 2
			}
			timeout = d
		case "-h", "--help":
			outln(stdout, "quikaitools generate --model DIR [--adapter DIR] | --gguf FILE [--prompt TEXT] [--template chatml|raw] [--messages JSON] [--tokens N] [--temperature F] [--top-k N] [--top-p F] [--seed N] [--stop-eos] [--timeout DURATION]")
			return 0
		default:
			outf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if template != "" && prompt == "" && messagesJSON == "" {
		outln(stderr, "generate --template requires --prompt or --messages")
		return 2
	}
	if messagesJSON != "" || template != "" {
		var msgs []chatfmt.Message
		if messagesJSON != "" {
			var err error
			msgs, err = chatfmt.ParseMessagesJSON(messagesJSON)
			if err != nil {
				outln(stderr, err)
				return 2
			}
		}
		formatted, err := chatfmt.Apply(template, msgs, prompt)
		if err != nil {
			outln(stderr, err)
			return 2
		}
		prompt = formatted
	} else if prompt == "" {
		prompt = "Hello"
	}
	if gguf != "" {
		if tempSet || topKSet || topPSet || stopEOS || seedSet {
			outln(stderr, "generate --gguf does not support --temperature, --top-k, --top-p, --seed, or --stop-eos")
			return 2
		}
		p, err := backend.Detect(profile, nil)
		if err != nil {
			outln(stderr, err)
			return 1
		}
		out, err := infer.GenerateGGUF(infer.GenerateGGUFOptions{Model: gguf, Prompt: prompt, Tokens: tokens, Profile: p, Timeout: timeout})
		if err != nil {
			outln(stderr, err)
			return 1
		}
		outln(stdout, out)
		return 0
	}
	if model == "" {
		outln(stderr, "generate requires --model or --gguf")
		return 2
	}
	base, err := gpt2.LoadDir(model)
	if err != nil {
		outln(stderr, err)
		return 1
	}
	tok, err := gpt2.LoadTokenizer(model)
	if err != nil {
		outln(stderr, err)
		return 1
	}
	var m *peft.Model
	if adapter != "" {
		m, err = peft.Load(base, adapter)
		if err != nil {
			outln(stderr, err)
			return 1
		}
	} else {
		m = peft.BaseOnly(base)
	}
	ids := tok.Encode(prompt)
	if len(ids) == 0 {
		ids = []int{0}
	}
	useSample := temperature > 0 || topKSet || topPSet || stopEOS
	if useSample {
		so := peft.SampleOptions{
			Temperature: temperature,
			TopK:        topK,
			TopP:        topP,
			EOS:         -1,
			Seed:        seed,
		}
		if stopEOS {
			so.EOS = tok.EOSID
		}
		outln(stdout, tok.Decode(m.Sample(ids, tokens, so)))
		return 0
	}
	outln(stdout, tok.Decode(m.Generate(ids, tokens)))
	return 0
}

func cmdEmbed(args []string, stdout, stderr io.Writer) int {
	visionMode := false
	allowStub := true
	var model, text, image, normalize string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--vision":
			visionMode = true
		case "--normalize":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			normalize = v
		case "--model":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			model = v
		case "--text":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			text = v
		case "--image":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			image = v
		case "--allow-stub", "--allow-stub=true", "--allow-stub=false":
			b, ok := boolFromFlag(args[i], stderr)
			if !ok {
				return 2
			}
			allowStub = b
		case "-h", "--help":
			outln(stdout, "quikaitools embed --model DIR --text TEXT | --vision --model DIR --image FILE [--normalize imagenet|clip] [--allow-stub]")
			return 0
		default:
			outf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if model == "" {
		outln(stderr, "embed requires --model")
		return 2
	}
	if visionMode {
		if image == "" {
			outln(stderr, "embed --vision requires --image")
			return 2
		}
		v, eng, err := infer.EmbedVision(infer.EmbedVisionOptions{ModelDir: model, Image: image, Size: 224, Normalize: normalize, StrictStub: !allowStub})
		if err != nil {
			outln(stderr, err)
			return 1
		}
		noteStubEngine(stderr, eng)
		outf(stdout, "engine=%s dim=%d\n", eng, len(v))
		outln(stdout, formatVec(v, 8))
		return 0
	}
	if text == "" {
		outln(stderr, "embed requires --text (or --vision --image)")
		return 2
	}
	v, eng, err := infer.EmbedText(infer.EmbedTextOptions{ModelDir: model, Text: text, StrictStub: !allowStub})
	if err != nil {
		outln(stderr, err)
		return 1
	}
	noteStubEngine(stderr, eng)
	outf(stdout, "engine=%s dim=%d\n", eng, len(v))
	outln(stdout, formatVec(v, 8))
	return 0
}

func cmdTranscribe(args []string, stdout, stderr io.Writer) int {
	var model, audioPath string
	allowStub := true
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--model":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			model = v
		case "--audio":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			audioPath = v
		case "--allow-stub", "--allow-stub=true", "--allow-stub=false":
			b, ok := boolFromFlag(args[i], stderr)
			if !ok {
				return 2
			}
			allowStub = b
		case "-h", "--help":
			outln(stdout, "quikaitools transcribe --model DIR --audio FILE [--allow-stub]")
			return 0
		default:
			outf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if model == "" || audioPath == "" {
		outln(stderr, "transcribe requires --model and --audio")
		return 2
	}
	text, eng, err := infer.Transcribe(infer.TranscribeOptions{ModelDir: model, Audio: audioPath, StrictStub: !allowStub})
	if err != nil {
		outln(stderr, err)
		return 1
	}
	noteStubEngine(stderr, eng)
	outf(stdout, "engine=%s\n%s\n", eng, text)
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
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			path = v
		case "-h", "--help":
			outln(stdout, "quikaitools validate-sft --path FILE.jsonl")
			return 0
		default:
			outf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if path == "" {
		outln(stderr, "validate-sft requires --path")
		return 2
	}
	total, issues, err := train.ValidateMessagesJSONL(path)
	if err != nil {
		outln(stderr, err)
		return 1
	}
	if len(issues) == 0 {
		outf(stdout, "VALID %s: %d rows\n", path, total)
		return 0
	}
	outf(stdout, "INVALID %s: %d rows, %d issues\n", path, total, len(issues))
	for i, issue := range issues {
		if i >= 20 {
			break
		}
		outf(stdout, "  - %s\n", issue)
	}
	return 1
}

func cmdExport(args []string, stdout, stderr io.Writer) int {
	if len(args) < 1 {
		outln(stderr, "export requires merge|gguf|modelfile")
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
		outln(stdout, "quikaitools export merge|gguf|modelfile …")
		return 0
	default:
		outf(stderr, "unknown export subcommand %q\n", sub)
		return 2
	}
}

func cmdExportMerge(args []string, stdout, stderr io.Writer) int {
	model, adapter, out := "", "", ""
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--model":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			model = v
		case "--adapter":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			adapter = v
		case "--out":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			out = v
		case "-h", "--help":
			outln(stdout, "quikaitools export merge --model DIR --adapter DIR --out DIR")
			return 0
		default:
			outf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if model == "" || adapter == "" || out == "" {
		outln(stderr, "export merge requires --model, --adapter, and --out")
		return 2
	}
	if err := exportx.MergeGPT2LoRA(model, adapter, out); err != nil {
		outln(stderr, err)
		return 1
	}
	outf(stdout, "merged -> %s\n", out)
	return 0
}

func cmdExportGGUF(args []string, stdout, stderr io.Writer) int {
	model, out, quant := "", "", "Q4_K_M"
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--model":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			model = v
		case "--out":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			out = v
		case "--quant":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			quant = v
		case "-h", "--help":
			outln(stdout, "quikaitools export gguf --model DIR --out FILE [--quant Q4_K_M]")
			return 0
		default:
			outf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if model == "" || out == "" {
		outln(stderr, "export gguf requires --model and --out")
		return 2
	}
	skip, err := exportx.ConvertGGUF(model, out, quant)
	if err != nil {
		outln(stderr, err)
		return 1
	}
	if skip != "" {
		outf(stdout, "SKIP: %s\n", skip)
		return 0
	}
	outf(stdout, "gguf -> %s\n", out)
	return 0
}

func cmdExportModelfile(args []string, stdout, stderr io.Writer) int {
	template, gguf, out := "", "", ""
	force := false
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--template":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			template = v
		case "--gguf":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			gguf = v
		case "--out":
			v, ok := nextArg(args, &i, stderr)
			if !ok {
				return 2
			}
			out = v
		case "--force", "--force=true", "--force=false":
			b, ok := boolFromFlag(args[i], stderr)
			if !ok {
				return 2
			}
			force = b
		case "-h", "--help":
			outln(stdout, "quikaitools export modelfile --gguf FILE --out FILE [--template FILE] [--force]")
			return 0
		default:
			outf(stderr, "unknown flag %s\n", args[i])
			return 2
		}
	}
	if gguf == "" || out == "" {
		outln(stderr, "export modelfile requires --gguf and --out")
		return 2
	}
	if err := exportx.WriteModelfileOpts(template, gguf, out, force); err != nil {
		outln(stderr, err)
		return 1
	}
	outf(stdout, "modelfile -> %s\n", out)
	return 0
}
