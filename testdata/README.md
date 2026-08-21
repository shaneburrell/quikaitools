# testdata

Checked-in fixtures live here (or next to the package under `internal/*/testdata`).

**Generated output goes in `artifacts/`.** That directory is gitignored. Do not commit coverage, benches, doctor dumps, or soak trees.

```bash
make cover    # testdata/artifacts/coverage.out + coverage.html + coverage.txt
make bench    # testdata/artifacts/bench.txt
make test     # also writes doctor-*.txt and catalog-*.txt from artifact tests
make clean    # removes testdata/artifacts
```

Override the output directory with `QUIKAITOOLS_ARTIFACTS` (used in tests via `internal/testart`).
