package main

import (
	"os"

	"github.com/shaneburrell/quikaitools/internal/cli"
)

func main() {
	os.Exit(cli.Main(os.Args, os.Stdout, os.Stderr))
}
