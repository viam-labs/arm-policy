RUNTIME_SOURCES := $(shell find src policy_server_runtime -path '*/.venv' -prune -o -type f -print)

module.tar.gz: meta.json pyproject.toml uv.lock run.sh $(RUNTIME_SOURCES)
	rm -f $@
	tar czf $@ --exclude='__pycache__' --exclude='*.pyc' --exclude='policy_server_runtime/.venv' meta.json pyproject.toml uv.lock run.sh src policy_server_runtime

test:
	uv run pytest tests/

clean:
	rm -rf module.tar.gz

.PHONY: test clean
