# jev-plays-pokemon
Like Claude Plays Pokemon, but with Jev

## Usage

Copy `.env.example` to `.env` and fill in at least `TYPESAFE_API_KEY`, then:

```
uv run jev-plays-pokemon              # same as `run`: boot the ROM and play
uv run jev-plays-pokemon run --rom-path pokemon_red.gb --stream-port 8080
uv run jev-plays-pokemon benchmark --turns 25 --warmup 5
uv run jev-plays-pokemon watchdog     # run `run` as a monitored, auto-restarting subprocess
```

Every `.env` value has a matching global CLI flag (e.g. `--openai-api-key`,
`--typesafe-api-key`) that overrides it for a single invocation - see
`uv run jev-plays-pokemon --help` and `docs/adr/0001-unified-typer-cli-and-config-settings.md`.

## Watching the run

`run` starts a small read-only HTTP server on `--stream-port` (default
`8080`) and logs its viewer URL at startup. It serves three endpoints:

- `/viewer` - the combined video + live decision sidebar page. **Open this
  in a browser, or use it as your OBS browser source.**
- `/video.mjpg` - the raw MJPEG video feed with no sidebar. Only needed for
  advanced/alternate OBS setups that want the video and decision data as
  separate sources.
- `/` - the latest decision as a raw JSON snapshot (action, confidence,
  milestones). This is by design, not a bug: it's meant for programmatic
  polling, not for opening in a browser.
