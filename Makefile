GO ?= go
BIN := bin/quikaitools

.PHONY: all build fmt vet test test-race tidy check clean

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

tidy:
	$(GO) mod tidy

check: tidy fmt vet test-race

clean:
	rm -rf bin dist testdata/artifacts
