# Workflow: Scene Cleanup

Remove floating artifacts ("floaters") and noise from trained gaussian splat scenes using LLM-guided selection.

## Via MCP (Agent-Controlled)

```bash
# Select floaters using natural language
lfs-mcp call selection.by_description \
  '{"description":"floating artifacts and noise outside the main object"}'

# Check what was selected
lfs-mcp call selection.get

# Delete selected gaussians
lfs-mcp call gaussians.write '{"delete_selected":true}'

# Verify with a render
lfs-mcp call render.capture '{"width":1920,"height":1080}'
```

## Crop Box Approach

```bash
# Add a crop box to define the region of interest
lfs-mcp call crop_box.add
lfs-mcp call crop_box.fit

# Adjust if needed
lfs-mcp call crop_box.set '{"center":[0,0,0],"size":[5,5,5]}'
```

## Ellipsoid Selection

```bash
# For roughly spherical subjects
lfs-mcp call ellipsoid.add
lfs-mcp call ellipsoid.fit

# Fine-tune
lfs-mcp call ellipsoid.set '{"center":[0,1,0],"radii":[2,3,2]}'
```

## Undo Safety

All operations support undo:

```bash
# Group operations into a transaction
lfs-mcp call history.begin_transaction '{"name":"cleanup"}'

# ... perform edits ...

# Commit or rollback
lfs-mcp call history.commit_transaction
# or: lfs-mcp call history.rollback_transaction
```
