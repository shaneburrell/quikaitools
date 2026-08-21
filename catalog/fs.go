package catalog

import "embed"

// Models is the shipped YAML catalog.
//
//go:embed models/*.yaml
var Models embed.FS
