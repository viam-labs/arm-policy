module.tar.gz: meta.json pyproject.toml uv.lock run.sh $(shell find src -name '*.py')
	rm -f $@
	tar czf $@ --exclude='__pycache__' --exclude='*.pyc' meta.json pyproject.toml uv.lock run.sh src

test:
	uv run pytest tests/

clean:
	rm -rf module.tar.gz

.PHONY: test clean
