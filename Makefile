module.tar.gz: meta.json uv.lock pyproject.toml run.sh $(git ls-files src)
	rm -f $@
	tar czf $@ $^
