# Model catalog

YAML files in [`models/`](models) are the zoo. Schema (fields, status values, `LoadDir` fallback): [docs/gaps/zoo.md](../docs/gaps/zoo.md).

```bash
quikaitools catalog
quikaitools catalog --task embed --machine v100
```

The same files are embedded in the binary via `catalog.Models` so `go install` still lists the starter set.
