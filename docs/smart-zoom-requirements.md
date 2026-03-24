# Smart Zoom — Plugin Requirements

This document defines the contract between reeln-cli core and the openai plugin
for the smart target zoom feature. The core handles frame extraction and filter
graph building; the plugin handles vision analysis.

## Overview

When a user renders a short with `--crop smart`, reeln-cli:

1. Extracts N frames from the input clip (via `Renderer.extract_frames`)
2. Emits `ON_FRAMES_EXTRACTED` hook with the frame paths and video metadata
3. Reads `context.shared["smart_zoom"]["zoom_path"]` after the hook returns
4. Builds a dynamic ffmpeg crop filter that pans across the detected targets

The plugin's job is step 2→3: analyze the frames and write the zoom path.

## Hook: ON_FRAMES_EXTRACTED

### When it fires

After frame extraction completes, before the render filter chain is built.
This hook fires during `reeln render short` and `reeln render preview` when
`crop_mode` is `smart`.

### context.data (read-only)

```python
from pathlib import Path
from reeln.models.zoom import ExtractedFrames

{
    "frames": ExtractedFrames(
        frame_paths=(Path("/tmp/.../frame_0001.png"), ...),
        timestamps=(1.0, 3.0, 5.0, 7.0, 9.0),
        source_width=1920,
        source_height=1080,
        duration=10.0,
        fps=59.94,
    ),
    "input_path": Path("/path/to/clip.mkv"),
    "crop_mode": "smart",
}
```

### context.shared (write)

The plugin writes zoom targets here. Core reads them after `emit()` returns.

```python
from reeln.models.zoom import ZoomPath, ZoomPoint

context.shared["smart_zoom"] = {
    "zoom_path": ZoomPath(
        points=(
            ZoomPoint(timestamp=1.0, center_x=0.3, center_y=0.4, confidence=0.95),
            ZoomPoint(timestamp=3.0, center_x=0.35, center_y=0.42, confidence=0.90),
            ZoomPoint(timestamp=5.0, center_x=0.5, center_y=0.45, confidence=0.88),
            ZoomPoint(timestamp=7.0, center_x=0.6, center_y=0.5, confidence=0.92),
            ZoomPoint(timestamp=9.0, center_x=0.7, center_y=0.48, confidence=0.91),
        ),
        source_width=1920,
        source_height=1080,
        duration=10.0,
    ),
}
```

### Optional: frame descriptions (dual use)

The same extracted frames can also be used to generate a clip description.
This is a separate concern from zoom targeting but shares the same hook.

```python
context.shared["frame_descriptions"] = {
    "descriptions": [
        "Player #7 carries the puck through the neutral zone",
        "Shot on goal from the left circle",
        "Goalie makes a glove save",
        "Rebound play in front of the net",
        "Puck cleared to the corner",
    ],
    "summary": "Fast break leading to a shot and save sequence",
}
```

This is a future extension — zoom is the priority. But design the handler
so adding description generation later is straightforward (separate feature flag,
separate API call, same frames).

## Data Models (from reeln-cli)

These are defined in `reeln/models/zoom.py` and imported by the plugin:

```python
from reeln.models.zoom import ExtractedFrames, ZoomPath, ZoomPoint
```

### ZoomPoint
| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `float` | Seconds into the clip |
| `center_x` | `float` | 0.0 (left) to 1.0 (right) |
| `center_y` | `float` | 0.0 (top) to 1.0 (bottom) |
| `confidence` | `float` | 0.0-1.0, detection confidence (default 1.0) |

### ZoomPath
| Field | Type | Description |
|-------|------|-------------|
| `points` | `tuple[ZoomPoint, ...]` | Ordered by timestamp |
| `source_width` | `int` | Original video width |
| `source_height` | `int` | Original video height |
| `duration` | `float` | Total clip duration in seconds |

### ExtractedFrames
| Field | Type | Description |
|-------|------|-------------|
| `frame_paths` | `tuple[Path, ...]` | PNG frame file paths |
| `timestamps` | `tuple[float, ...]` | Timestamp of each frame |
| `source_width` | `int` | Video width |
| `source_height` | `int` | Video height |
| `duration` | `float` | Clip duration |
| `fps` | `float` | Video frame rate |

## Plugin Implementation

### New config fields

```python
ConfigField(
    name="smart_zoom_enabled",
    field_type="bool",
    default=False,
    description="Enable smart zoom target detection via OpenAI vision",
),
ConfigField(
    name="smart_zoom_model",
    field_type="str",
    default="gpt-4.1",
    description="OpenAI model for smart zoom vision analysis",
),
```

Prompt override is already supported via the existing `prompt_overrides` config.

### New hook registration

```python
def register(self, registry: HookRegistry) -> None:
    registry.register(Hook.ON_GAME_INIT, self.on_game_init)
    registry.register(Hook.ON_FRAMES_EXTRACTED, self.on_frames_extracted)
```

### Handler skeleton

```python
def on_frames_extracted(self, context: HookContext) -> None:
    """Analyze extracted frames for zoom targets."""
    if not self._config.get("smart_zoom_enabled", False):
        return

    frames: ExtractedFrames = context.data["frames"]
    client = self._get_client()

    points: list[ZoomPoint] = []
    for frame_path, timestamp in zip(frames.frame_paths, frames.timestamps):
        center_x, center_y = analyze_frame_for_zoom(
            client=client,
            prompt_registry=self._prompt_registry,
            frame_path=frame_path,
            model=str(self._config.get("smart_zoom_model", "gpt-4.1")),
        )
        points.append(ZoomPoint(
            timestamp=timestamp,
            center_x=center_x,
            center_y=center_y,
        ))

    zoom_path = ZoomPath(
        points=tuple(points),
        source_width=frames.source_width,
        source_height=frames.source_height,
        duration=frames.duration,
    )
    context.shared["smart_zoom"] = {"zoom_path": zoom_path}
```

### New module: `zoom.py`

```python
def analyze_frame_for_zoom(
    client: OpenAIClient,
    prompt_registry: PromptRegistry,
    frame_path: Path,
    *,
    model: str = "gpt-4.1",
) -> tuple[float, float]:
    """Send a frame to OpenAI vision and get zoom target coordinates.

    Returns (center_x, center_y) as normalized 0.0-1.0 floats.
    Falls back to (0.5, 0.5) on failure.
    """
```

Uses `client.request_structured()` with the frame as a base64 image input.

### New prompt template: `prompt_templates/smart_zoom_detect.txt`

```
You are analyzing a frame from a video to identify the main area of action.

Task:
- Identify the primary subject or action area in the frame.
- Return the center of that area as normalized coordinates (0.0-1.0).
- 0.0, 0.0 is the top-left corner; 1.0, 1.0 is the bottom-right corner.
- If no clear action area is visible, return the center of the frame.

Return JSON only:
{"center_x": 0.5, "center_y": 0.5}
```

### Response schema

```python
ZOOM_SCHEMA = {
    "type": "object",
    "properties": {
        "center_x": {"type": "number"},
        "center_y": {"type": "number"},
    },
    "required": ["center_x", "center_y"],
    "additionalProperties": False,
}
```

### Error handling

- API failure on a single frame: log warning, use `(0.5, 0.5)` fallback
- API failure on all frames: log warning, do NOT write to `context.shared`
  (core will fall back to static center crop)
- Clamp returned values to 0.0-1.0

## Testing Requirements

- Mock `OpenAIClient` — never call real API in tests
- Test feature flag gating (disabled → no API calls, no shared writes)
- Test with 1 frame, 3 frames, 5 frames
- Test API failure fallback per-frame
- Test API failure on all frames (no shared write)
- Test coordinate clamping (API returns out-of-range values)
- Test prompt template rendering and override
- 100% line + branch coverage

## Registry Update

Add `hook:ON_FRAMES_EXTRACTED` to the openai plugin entry in
`reeln-cli/registry/plugins.json`:

```json
{
    "name": "openai",
    "capabilities": [
        "hook:ON_GAME_INIT",
        "hook:ON_FRAMES_EXTRACTED"
    ]
}
```

## Config Profile Example

A user enables smart zoom via their reeln config profile:

```json
{
    "render_profiles": {
        "smart_replay": {
            "crop_mode": "smart",
            "smart_zoom_frames": 5
        }
    },
    "plugins": {
        "settings": {
            "openai": {
                "enabled": true,
                "smart_zoom_enabled": true,
                "smart_zoom_model": "gpt-4.1"
            }
        }
    }
}
```

Usage: `reeln render short clip.mkv --render-profile smart_replay`

## Phasing

This plugin work (Phase 6) can start as soon as Phase 1 (contracts/models) lands
in reeln-cli. The plugin imports `ZoomPoint`, `ZoomPath`, and `ExtractedFrames`
from `reeln.models.zoom`, so that module must exist first.

For local development before Phase 1 merges, you can copy the dataclass
definitions into a local stub or install reeln-cli from the feature branch.
