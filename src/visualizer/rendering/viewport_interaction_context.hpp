/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/rendering.hpp"
#include "rendering_types.hpp"

namespace lfs::vis {

    class SceneManager;

    struct ViewportInteractionContext {
        SceneManager* scene_manager = nullptr;
        lfs::rendering::ViewportData viewport_data{};
        lfs::rendering::ViewportData secondary_viewport_data{};
        ViewportRegion viewport_region{};
        bool pick_context_valid = false;
        bool secondary_viewport_valid = false;
        bool independent_split_active = false;
        float split_position = 0.5f;

        void updatePickContext(const ViewportRegion* region,
                               const lfs::rendering::ViewportData& data,
                               const lfs::rendering::ViewportData* secondary_data = nullptr,
                               const bool independent_split = false,
                               const float split_pos = 0.5f) {
            if (region) {
                viewport_region = *region;
                viewport_data = data;
                if (secondary_data) {
                    secondary_viewport_data = *secondary_data;
                    secondary_viewport_valid = true;
                } else {
                    secondary_viewport_data = {};
                    secondary_viewport_valid = false;
                }
                independent_split_active = independent_split;
                split_position = split_pos;
                pick_context_valid = true;
            } else {
                pick_context_valid = false;
                secondary_viewport_valid = false;
                independent_split_active = false;
            }
        }
    };

} // namespace lfs::vis
