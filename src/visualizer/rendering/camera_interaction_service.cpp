/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "camera_interaction_service.hpp"
#include "rendering/rendering.hpp"
#include "scene/scene_manager.hpp"
#include <algorithm>

namespace lfs::vis {

    int CameraInteractionService::pickCameraFrustum(
        lfs::rendering::RenderingEngine* const engine,
        SceneManager* const scene_manager,
        const ViewportInteractionContext& viewport_context,
        const RenderSettings& settings,
        const glm::vec2& mouse_pos,
        bool& hover_changed) {
        hover_changed = false;

        if (!settings.show_camera_frustums) {
            return -1;
        }

        const auto now = std::chrono::steady_clock::now();
        if (shouldThrottlePick(now)) {
            return hovered_camera_id_;
        }
        notePick(now);

        if (!engine || !scene_manager || !viewport_context.pick_context_valid) {
            return hovered_camera_id_;
        }

        auto cameras = scene_manager->getScene().getVisibleCameras();
        if (cameras.empty()) {
            return -1;
        }

        const lfs::rendering::ViewportData* viewport_data = &viewport_context.viewport_data;
        glm::vec2 viewport_pos(viewport_context.viewport_region.x, viewport_context.viewport_region.y);
        glm::vec2 viewport_size(viewport_context.viewport_region.width, viewport_context.viewport_region.height);
        if (viewport_context.independent_split_active &&
            viewport_context.secondary_viewport_valid &&
            viewport_context.viewport_region.width > 1.0f) {
            const auto layouts = makeSplitViewPanelLayouts(
                std::max(static_cast<int>(viewport_context.viewport_region.width), 1),
                viewport_context.split_position);
            const float local_x = mouse_pos.x - viewport_context.viewport_region.x;
            const SplitViewPanelId panel = local_x >= static_cast<float>(layouts[0].width)
                                               ? SplitViewPanelId::Right
                                               : SplitViewPanelId::Left;
            const size_t panel_index = splitViewPanelIndex(panel);
            viewport_pos.x += static_cast<float>(layouts[panel_index].x);
            viewport_size.x = static_cast<float>(layouts[panel_index].width);
            viewport_data = (panel == SplitViewPanelId::Right)
                                ? &viewport_context.secondary_viewport_data
                                : &viewport_context.viewport_data;
        }

        glm::mat4 scene_transform(1.0f);
        const auto transforms = scene_manager->getScene().getVisibleNodeTransforms();
        if (!transforms.empty()) {
            scene_transform = transforms[0];
        }

        const lfs::rendering::CameraFrustumPickRequest request{
            .mouse_pos = mouse_pos,
            .viewport_pos = viewport_pos,
            .viewport_size = viewport_size,
            .viewport = *viewport_data,
            .scale = settings.camera_frustum_scale,
            .scene_transform = scene_transform};

        const auto pick_result = engine->pickCameraFrustum(cameras, request);

        int cam_id = -1;
        if (pick_result) {
            cam_id = *pick_result;
        }

        hover_changed = updateHoveredCamera(cam_id);
        return hovered_camera_id_;
    }

} // namespace lfs::vis
