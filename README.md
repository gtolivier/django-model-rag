# django-model-rag

Discover and normalize text content from any Django model, ready for a RAG
pipeline.

**It is** a registry and a set of extractors that turn instances of your
Django models into plain-text `NormalizedDocument`s — with a title, a
language, a stable source key and a resolved URL — so that a retrieval
pipeline can index them and cite real pages.

**It is not** a RAG pipeline. It does not chunk, embed, store vectors or call
an LLM. That is the job of a separate package,
[django-minimal-rag](https://github.com/gtolivier/django-minimal-rag), which
does not depend on this one.

## Status

Pre-alpha. Nothing is implemented yet: the package is being written
test-first, using an earlier prototype as its behavioral specification. See
the [roadmap](ROADMAP.md) for the order of the features.

## Requirements

- Python 3.11+
- Django 5.2 LTS, 6.0 or 6.1

## Development

```sh
uv sync
uv run pytest
uv run ruff check
uv run ruff format --check
```

## License

MIT — see [LICENSE](LICENSE).
