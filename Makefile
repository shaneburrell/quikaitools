GO ?= go
BIN := bin/quikaitools
COVER_PKG := ./internal/...
COVER_MIN ?= 70
ARTIFACTS := testdata/artifacts

.PHONY: all build fmt vet test test-race cover bench tidy check clean

all: check build

build:
	mkdir -p bin
	$(GO) build -o $(BIN) ./cmd/quikaitools

fmt:
	$(GO) fmt ./...

vet:
	$(GO) vet ./...

test:
	$(GO) test ./...

test-race:
	$(GO) test -race ./...

cover:
	mkdir -p $(ARTIFACTS)
	$(GO) test $(COVER_PKG) -coverprofile=$(ARTIFACTS)/coverage.out -covermode=atomic
	$(GO) tool cover -html=$(ARTIFACTS)/coverage.out -o $(ARTIFACTS)/coverage.html
	$(GO) tool cover -func=$(ARTIFACTS)/coverage.out | tee $(ARTIFACTS)/coverage.txt
	@total=$$($(GO) tool cover -func=$(ARTIFACTS)/coverage.out | awk '/^total:/{print $$3}' | tr -d '%'); \
	echo "total coverage: $${total}% (min $(COVER_MIN)%)"; \
	awk -v t="$$total" -v m="$(COVER_MIN)" 'BEGIN{ if (t+0 < m+0) { printf("coverage %.1f%% is below %s%% gate\n", t, m); exit 1 } }'

bench:
	mkdir -p $(ARTIFACTS)
	$(GO) test -bench=. -benchmem -count=1 ./internal/... | tee $(ARTIFACTS)/bench.txt

tidy:
	$(GO) mod tidy

check: tidy fmt vet test-race cover

clean:
	rm -rf bin dist testdata/artifacts
