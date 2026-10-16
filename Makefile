RUNTIME_SOURCES := $(shell find src policy_server_runtime -path '*/.venv' -prune -o -name '__pycache__' -prune -o -type f ! -name '*.pyc' -print)
# The registry entrypoint is generated, not committed: Viam cloud build treats a
# module whose entrypoint already exists in source as pre-built and skips `make`.
ENTRYPOINT := dist/arm-policy

module.tar.gz: meta.json pyproject.toml uv.lock run.sh .python-version $(ENTRYPOINT) $(RUNTIME_SOURCES)
	rm -f $@
	tar czf $@ --exclude='__pycache__' --exclude='**/*.pyc' --exclude='.venv' $^

$(ENTRYPOINT): run.sh
	mkdir -p $(@D)
	printf '#!/usr/bin/env bash\nexec "$$(dirname "$$0")/../run.sh" "$$@"\n' > $@
	chmod +x $@

lint:
	uv run --with ruff ruff check .

test:
	uv run pytest tests/

clean:
	rm -rf module.tar.gz $(ENTRYPOINT)

.PHONY: test clean
