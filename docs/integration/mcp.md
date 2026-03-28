# MCP Server Integration

LichtFeld Studio includes a production-grade Model Context Protocol (MCP) server with 70+ tools, enabling programmatic control from Claude Code, Claude Desktop, or any MCP-compatible client.

## Connection

| Property | Value |
|----------|-------|
| **Protocol** | JSON-RPC 2.0 |
| **Protocol Version** | 2024-11-05 |
| **Transport** | HTTP POST |
| **Endpoint** | `http://127.0.0.1:45677/mcp` |
| **Methods** | `initialize`, `ping`, `tools/list`, `tools/call`, `resources/list`, `resources/read` |

## Setup for Claude Code

### Option A: MCP Server Config (auto-launches the app)

```json
{
  "mcpServers": {
    "lichtfeld": {
      "command": "python3",
      "args": ["scripts/lichtfeld_mcp_bridge.py"],
      "env": {
        "LICHTFELD_EXECUTABLE": "/path/to/build/LichtFeld-Studio",
        "LD_LIBRARY_PATH": "/path/to/build"
      }
    }
  }
}
```

### Option B: CLI Registration

```bash
claude mcp add lichtfeld -s user -- python3 /path/to/scripts/lichtfeld_mcp_bridge.py
```

### Option C: Direct HTTP (when app is already running)

```bash
curl -s -X POST http://127.0.0.1:45677/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

## Tool Inventory (70+ tools)

### Training Control

| Tool | Description |
|------|-------------|
| `scene.load_dataset` | Load COLMAP dataset for training |
| `scene.load_checkpoint` | Resume from checkpoint file |
| `scene.save_checkpoint` | Save training state |
| `training.start` | Begin/resume training |
| `training.get_state` | Current iteration, loss, gaussian count |
| `training.get_loss_history` | Loss curve data points |
| `training.ask_advisor` | LLM-based training advice |

### Camera

| Tool | Description |
|------|-------------|
| `camera.get` | Current camera position/rotation/FOV |
| `camera.set_view` | Set camera transform |
| `camera.reset` | Reset to default |
| `camera.list` | List dataset cameras |
| `camera.go_to_dataset_camera` | Jump to dataset camera by index |

### Rendering

| Tool | Description |
|------|-------------|
| `render.capture` | Render to base64 PNG |
| `render.settings.get` | Current render settings |
| `render.settings.set` | Modify render settings |

### Selection

| Tool | Description |
|------|-------------|
| `selection.rect` | Rectangle selection |
| `selection.polygon` | Polygon selection |
| `selection.lasso` | Freeform lasso |
| `selection.ring` | Pick single gaussian |
| `selection.brush` | Radius selection |
| `selection.click` | Click selection |
| `selection.get` | Return selected indices |
| `selection.clear` | Clear selection |
| `selection.by_description` | Natural language selection (LLM vision) |

### Export

| Tool | Description |
|------|-------------|
| `scene.export_ply` | Export as PLY |
| `scene.export_sog` | Export as SOG |
| `scene.export_spz` | Export as SPZ (compressed) |
| `scene.export_usd` | Export as USD |
| `scene.export_html` | Self-contained HTML viewer |
| `scene.export_status` | Check async export progress |

### Scene Graph

| Tool | Description |
|------|-------------|
| `scene.list_nodes` | List scene tree |
| `scene.select_node` | Select node by name |
| `scene.set_node_visibility` | Toggle visibility |
| `scene.rename_node` | Rename node |
| `scene.add_group` | Create group |
| `scene.duplicate_node` | Duplicate |
| `scene.merge_group` | Merge children |

### History (Undo/Redo)

| Tool | Description |
|------|-------------|
| `history.undo` / `history.redo` | Undo/redo |
| `history.begin_transaction` | Start grouped operation |
| `history.commit_transaction` | Commit group |
| `history.rollback_transaction` | Rollback |

### Python Editor

| Tool | Description |
|------|-------------|
| `editor.set_code` | Set Python code |
| `editor.run` | Execute |
| `editor.get_output` | Read stdout/stderr |
| `editor.wait` | Wait for completion |
| `editor.interrupt` | Kill script |

### Events (Pub/Sub)

| Tool | Description |
|------|-------------|
| `events.subscribe` | Subscribe to event type |
| `events.poll` | Poll for events |
| `events.unsubscribe` | Unsubscribe |
| `events.list` | List event types |

### Low-Level

| Tool | Description |
|------|-------------|
| `gaussians.read` | Read GPU tensor data |
| `gaussians.write` | Write GPU tensor data |
| `plugin.invoke` | Invoke plugin capability |
| `plugin.list` | List plugins |

## Resources (Read-Only)

| URI | Description |
|-----|-------------|
| `lichtfeld://training/state` | Training snapshot |
| `lichtfeld://training/loss_curve` | Loss history |
| `lichtfeld://render/current` | Viewport as base64 PNG |
| `lichtfeld://scene/nodes` | Scene graph |
| `lichtfeld://selection/mask` | Selection mask |
| `lichtfeld://history/state` | Undo/redo state |
| `lichtfeld://editor/code` | Editor content |
| `lichtfeld://editor/output` | Script output |

## lfs-mcp CLI

A convenience wrapper for the HTTP API:

```bash
lfs-mcp ping                                    # Health check
lfs-mcp list                                     # List all tools
lfs-mcp resources                                # List all resources
lfs-mcp call training.get_state                  # Call a tool
lfs-mcp call render.capture '{"width":1920}'     # With arguments
lfs-mcp read lichtfeld://training/state          # Read a resource
```

## Known Limitation: Headless Mode

The `--headless` flag does not currently start the MCP server. MCP is only available in GUI mode. A ~5 line patch to `src/app/application.cpp:runHeadless()` would enable it. See [Headless MCP Gap](../troubleshooting/headless-mcp.md).
