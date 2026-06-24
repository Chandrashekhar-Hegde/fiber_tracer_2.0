# Fiber Tracer Terminal UI

A keyboard-driven, themeable terminal interface for the Fiber Tracer RAFA analysis pipeline.

## Requirements

- [Bun](https://bun.sh/) ≥ 1.2
- [Node.js](https://nodejs.org/) ≥ 18

## Install

```bash
cd tui
bun install
```

## Development

```bash
bun run dev
```

## Build

```bash
bun run build
```

## Type check

```bash
bun run typecheck
```

## Test

```bash
bun test
```

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `1` | New Analysis wizard |
| `2` | Dashboard |
| `3` | History |
| `4` | Experiments |
| `5` | Model Registry |
| `6` | Training |
| `7` | Logs |
| `8` | Settings |
| `←` / `p` | Previous wizard step |
| `→` / `n` | Next wizard step |
| `Enter` | Select / confirm in wizard |
| `r` | Run analysis from the Review step |
| `q` | Quit |

## Themes

Themes are configured in `~/.config/fiber-tracer/config.json`. The bundled themes are `dracula`, `catppuccin-mocha`, `catppuccin-latte`, `one-dark`, and `nord`.

## Project documentation

See the main project [`README.md`](../README.md) for installation, usage, and development instructions.
