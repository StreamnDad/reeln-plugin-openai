<!--
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Find-and-replace these placeholders before using this template:    │
  │                                                                    │
  │   plugin_name   → Python package name    (e.g., reeln_google_plugin)│
  │   plugin-name   → PyPI/repo name         (e.g., reeln-plugin-google)│
  │   PluginName    → Plugin class name      (e.g., GooglePlugin)      │
  │   PLUGIN_PKG    → Makefile/CI variable   (same as plugin_name)     │
  │   myplugin      → Plugin short name      (e.g., google)           │
  │                                                                    │
  │ Files to update: pyproject.toml, Makefile, CI workflows,           │
  │   plugin_name/__init__.py, plugin_name/plugin.py,                  │
  │   tests/unit/test_plugin.py, CLAUDE.md                             │
  │                                                                    │
  │ Also rename the plugin_name/ directory to your package name.       │
  └──────────────────────────────────────────────────────────────────────┘
-->

# plugin-name

A [reeln-cli](https://github.com/StreamnDad/reeln-cli) plugin for ...

## Install

```bash
pip install plugin-name
```

Or for development:

```bash
git clone https://github.com/StreamnDad/plugin-name
cd plugin-name
make dev-install
```

## Configuration

Add the plugin to your reeln config:

```json
{
  "plugins": {
    "enabled": ["myplugin"],
    "settings": {
      "myplugin": {}
    }
  }
}
```

## Development

```bash
make dev-install    # uv venv + editable install with dev deps
make test           # pytest with 100% coverage
make lint           # ruff check
make format         # ruff format
make check          # lint + mypy + test
```

## License

AGPL-3.0-only
