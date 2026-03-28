# Known Issue: Headless Mode Does Not Start MCP Server

## Problem

Running LichtFeld Studio with `--headless` does not start the MCP HTTP server on port 45677. MCP is currently only available in GUI mode.

## Root Cause

In `src/app/application.cpp`, the `runGui()` path (line 288-302) initialises MCP:

```cpp
mcp::register_core_tools();
mcp::register_core_resources();
register_gui_scene_tools(viewer.get());
register_gui_scene_resources(viewer.get());

mcp::McpHttpServer mcp_http({.enable_resources = true});
mcp_http.start();
```

The `runHeadless()` path (line 46-200) skips all MCP initialisation.

## Workarounds

1. **Use GUI mode** for MCP access (the MCP bridge auto-launches in GUI mode)
2. **Use CLI flags** for headless training (no MCP needed)
3. **Apply the patch below** to enable MCP in headless mode

## Patch

Insert after line 53 in `runHeadless()`:

```cpp
// Start MCP server in headless mode
mcp::register_core_tools();
mcp::register_core_resources();
// Note: headless-specific tools from mcp_training_context.cpp
// are already registered via register_builtin_tools()
mcp::register_builtin_tools();

mcp::McpHttpServer mcp_http({.enable_resources = true});
if (!mcp_http.start())
    LOG_ERROR("Failed to start MCP HTTP server in headless mode");
```

And before the function returns, add:

```cpp
mcp_http.stop();
```

## Impact

Without this patch, agent-controlled headless training requires launching in GUI mode or using CLI arguments directly. The `video2splat` pipeline works around this by using CLI flags rather than MCP.
