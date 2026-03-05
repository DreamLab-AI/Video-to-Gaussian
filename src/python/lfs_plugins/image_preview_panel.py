# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Image preview panel using RmlUI floating window."""

from math import gcd
from pathlib import Path
from typing import Optional

import lichtfeld as lf
from .types import RmlPanel

ZOOM_MIN = 0.1
ZOOM_MAX = 10.0

# RmlUI key identifiers (Rml::Input::KeyIdentifier)
KI_SPACE = 1
KI_1 = 3
KI_F = 17
KI_I = 20
KI_M = 24
KI_R = 29
KI_T = 31
KI_OEM_PLUS = 39
KI_OEM_MINUS = 41
KI_ESCAPE = 81
KI_END = 88
KI_HOME = 89
KI_LEFT = 90
KI_UP = 91
KI_RIGHT = 92
KI_DOWN = 93

_instance = None


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


class ImagePreviewPanel(RmlPanel):
    idname = "lfs.image_preview"
    label = "Image Preview"
    space = "FLOATING"
    order = 98
    rml_template = "rmlui/image_preview.rml"
    initial_width = 900
    initial_height = 600

    def __init__(self):
        global _instance
        _instance = self

        self._image_paths: list[Path] = []
        self._mask_paths: list[Optional[Path]] = []
        self._current_index = 0

        self._zoom = 1.0
        self._fit_to_window = True
        self._show_info = True
        self._show_filmstrip = True
        self._show_overlay = False

        self._doc = None
        self._dirty = True
        self._filmstrip_needs_rebuild = True
        self._prev_index = -1

        self._image_info_cache: dict[str, tuple[int, int, int]] = {}

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("image_preview")
        if model is None:
            return

        tr = lf.ui.tr
        model.bind_func("panel_label", lambda: tr("image_preview.title"))
        self._handle = model.get_handle()

    def on_load(self, doc):
        super().on_load(doc)
        self._doc = doc

        btn_fit = doc.get_element_by_id("btn-fit")
        if btn_fit:
            btn_fit.add_event_listener("click", lambda _ev: self._set_fit_to_window())

        cb_fit = doc.get_element_by_id("cb-fit")
        if cb_fit:
            cb_fit.add_event_listener("change", self._on_fit_checkbox_change)

        cb_mask = doc.get_element_by_id("cb-mask")
        if cb_mask:
            cb_mask.add_event_listener("change", self._on_mask_checkbox_change)

        wf = doc.get_element_by_id("window-frame")

        doc.add_event_listener("keydown", self._on_keydown)
        if wf:
            wf.add_event_listener("keydown", self._on_keydown)

        self._dirty = True

    def on_update(self, doc):
        if not self._dirty:
            return
        self._dirty = False
        self._refresh_ui(doc)

    def open(self, image_paths: list[Path], mask_paths: list[Optional[Path]], start_index: int):
        if not image_paths:
            return

        self._image_paths = [p.resolve() for p in image_paths]
        self._mask_paths = [p.resolve() if p else None for p in mask_paths] if mask_paths else [None] * len(image_paths)
        self._current_index = min(start_index, len(image_paths) - 1)
        self._zoom = 1.0
        self._fit_to_window = True
        self._dirty = True
        self._filmstrip_needs_rebuild = True
        self._prev_index = -1

    def _navigate(self, delta: int):
        new_idx = self._current_index + delta
        if 0 <= new_idx < len(self._image_paths):
            self._current_index = new_idx
            self._dirty = True

    def _go_to_image(self, index: int):
        if 0 <= index < len(self._image_paths):
            self._current_index = index
            self._dirty = True

    def _set_fit_to_window(self):
        self._fit_to_window = True
        self._zoom = 1.0
        self._dirty = True

    def _has_valid_overlay(self) -> bool:
        if self._current_index >= len(self._mask_paths):
            return False
        mask_path = self._mask_paths[self._current_index]
        return mask_path is not None and mask_path.exists()

    def _close_panel(self):
        lf.ui.set_panel_enabled("lfs.image_preview", False)

    def _get_image_info(self, path: Path) -> tuple[int, int, int]:
        key = str(path)
        if key not in self._image_info_cache:
            try:
                self._image_info_cache[key] = lf.ui.get_image_info(key)
            except Exception:
                self._image_info_cache[key] = (0, 0, 0)
        return self._image_info_cache[key]

    @staticmethod
    def _format_aspect_ratio(w: int, h: int) -> str:
        if w <= 0 or h <= 0:
            return ""
        d = gcd(w, h)
        rw, rh = w // d, h // d
        if rw > 30 or rh > 30:
            ratio = w / h
            common = {
                (16, 9), (16, 10), (4, 3), (3, 2), (21, 9),
                (1, 1), (5, 4), (3, 1), (2, 1),
            }
            best = min(common, key=lambda r: abs(r[0] / r[1] - ratio))
            if abs(best[0] / best[1] - ratio) < 0.05:
                return f"{best[0]}:{best[1]}"
            return f"{w / h:.2f}:1"
        return f"{rw}:{rh}"

    def _on_fit_checkbox_change(self, event):
        cb = self._doc.get_element_by_id("cb-fit") if self._doc else None
        if cb:
            self._fit_to_window = cb.has_attribute("checked")
            if self._fit_to_window:
                self._zoom = 1.0
            self._dirty = True

    def _on_mask_checkbox_change(self, event):
        cb = self._doc.get_element_by_id("cb-mask") if self._doc else None
        if cb:
            self._show_overlay = cb.has_attribute("checked")
            self._dirty = True

    def _apply_zoom(self, img_el, path: Path):
        if self._fit_to_window:
            img_el.remove_property("width")
            img_el.remove_property("height")
            return
        w, h, _ = self._get_image_info(path)
        if w <= 0 or h <= 0:
            return
        dw = int(w * self._zoom)
        dh = int(h * self._zoom)
        img_el.set_property("width", f"{dw}dp")
        img_el.set_property("height", f"{dh}dp")

    # -- UI refresh --

    def _refresh_ui(self, doc):
        has_images = bool(self._image_paths)

        self._update_main_image(doc, has_images)
        self._update_filmstrip(doc, has_images)
        self._update_sidebar(doc, has_images)
        self._update_status(doc, has_images)

    def _update_main_image(self, doc, has_images: bool):
        main_img = doc.get_element_by_id("main-image")
        mask_img = doc.get_element_by_id("mask-overlay")
        no_text = doc.get_element_by_id("no-image-text")

        if not has_images:
            if main_img:
                main_img.set_attribute("class", "hidden")
                main_img.set_attribute("src", "")
            if mask_img:
                mask_img.set_attribute("class", "")
            if no_text:
                no_text.set_attribute("class", "")
                no_text.set_inner_rml(_xml_escape(lf.ui.tr("image_preview.no_images_loaded")))
            return

        path = self._image_paths[self._current_index]
        if main_img:
            main_img.set_attribute("class", "")
            main_img.set_attribute("src", str(path))
            self._apply_zoom(main_img, path)
        if no_text:
            no_text.set_attribute("class", "hidden")

        show_mask = self._show_overlay and self._has_valid_overlay()
        if mask_img:
            if show_mask:
                mask_path = self._mask_paths[self._current_index]
                mask_img.set_attribute("src", str(mask_path))
                mask_img.set_attribute("class", "visible")
            else:
                mask_img.set_attribute("class", "")
                mask_img.set_attribute("src", "")

    def _update_filmstrip(self, doc, has_images: bool):
        filmstrip = doc.get_element_by_id("filmstrip")
        if not filmstrip:
            return

        if not self._show_filmstrip:
            filmstrip.set_attribute("class", "hidden")
            return
        filmstrip.set_attribute("class", "")

        if not has_images:
            filmstrip.set_inner_rml("")
            self._filmstrip_needs_rebuild = True
            return

        if self._filmstrip_needs_rebuild:
            self._filmstrip_needs_rebuild = False
            parts = []
            for i in range(len(self._image_paths)):
                cls = "thumb-item selected" if i == self._current_index else "thumb-item"
                parts.append(f'<div class="{cls}" id="thumb-{i}"></div>')
            filmstrip.set_inner_rml("\n".join(parts))

            for i, path in enumerate(self._image_paths):
                el = doc.get_element_by_id(f"thumb-{i}")
                if el:
                    el.set_property("decorator", f"image({path})")
                    el.add_event_listener("click", lambda _ev, idx=i: self._go_to_image(idx))
            self._scroll_filmstrip(filmstrip, self._current_index)
            self._prev_index = self._current_index
        elif self._prev_index != self._current_index:
            if self._prev_index >= 0:
                old_el = doc.get_element_by_id(f"thumb-{self._prev_index}")
                if old_el:
                    old_el.set_attribute("class", "thumb-item")
            new_el = doc.get_element_by_id(f"thumb-{self._current_index}")
            if new_el:
                new_el.set_attribute("class", "thumb-item selected")
            self._scroll_filmstrip(filmstrip, self._current_index)
            self._prev_index = self._current_index

    def _update_sidebar(self, doc, has_images: bool):
        sidebar = doc.get_element_by_id("sidebar")
        if not sidebar:
            return

        if not self._show_info:
            sidebar.set_attribute("class", "hidden")
            return
        sidebar.set_attribute("class", "")

        tr = lf.ui.tr

        _set_text(doc, "sidebar-image-label", tr("image_preview.image_section"))
        _set_text(doc, "sidebar-file-label", tr("image_preview.file_section"))
        _set_text(doc, "sidebar-view-label", tr("image_preview.view_section"))

        if has_images:
            path = self._image_paths[self._current_index]
            ext = path.suffix[1:].upper() if path.suffix else "?"
            w, h, _c = self._get_image_info(path)

            if w > 0 and h > 0:
                _set_text(doc, "sidebar-dimensions", f"{w} x {h}")
                mp = (w * h) / 1_000_000
                ratio = self._format_aspect_ratio(w, h)
                mp_text = f"{mp:.1f} MP"
                if ratio:
                    mp_text += f" \u00b7 {ratio}"
                _set_text(doc, "sidebar-megapixels", mp_text)
            else:
                _set_text(doc, "sidebar-dimensions", "")
                _set_text(doc, "sidebar-megapixels", "")

            if path.exists():
                size_bytes = path.stat().st_size
                size_str = self._format_size(size_bytes)
                _set_text(doc, "sidebar-fileinfo", f"{size_str} \u00b7 {ext}")
            else:
                _set_text(doc, "sidebar-fileinfo", ext)

            parent_str = str(path.parent)
            if len(parent_str) > 25:
                parent_str = "..." + parent_str[-22:]
            _set_text(doc, "sidebar-filepath", parent_str)
        else:
            _set_text(doc, "sidebar-fileinfo", "")
            _set_text(doc, "sidebar-filepath", "")
            _set_text(doc, "sidebar-dimensions", "")
            _set_text(doc, "sidebar-megapixels", "")

        zoom_text = f"{tr('image_preview.zoom')}: {self._get_zoom_display()}"
        _set_text(doc, "sidebar-zoom", zoom_text)

        cb_fit = doc.get_element_by_id("cb-fit")
        if cb_fit:
            if self._fit_to_window:
                cb_fit.set_attribute("checked", "")
            else:
                cb_fit.remove_attribute("checked")
        _set_text(doc, "cb-fit-label", tr("image_preview.fit_to_window"))

        has_mask = self._has_valid_overlay()
        mask_sep = doc.get_element_by_id("sidebar-mask-sep")
        mask_label = doc.get_element_by_id("sidebar-mask-label")
        mask_name = doc.get_element_by_id("sidebar-mask-name")
        mask_row = doc.get_element_by_id("mask-check-row")

        if has_mask:
            if mask_sep:
                mask_sep.set_attribute("class", "separator")
            if mask_label:
                mask_label.set_inner_rml(_xml_escape(tr("image_preview.mask_section")))
                mask_label.set_attribute("class", "sidebar-section")
            if mask_name:
                name = self._mask_paths[self._current_index].name
                mask_name.set_inner_rml(_xml_escape(name))
                mask_name.set_attribute("class", "sidebar-value")
            if mask_row:
                mask_row.set_attribute("class", "sidebar-check-row")
            cb_mask = doc.get_element_by_id("cb-mask")
            if cb_mask:
                if self._show_overlay:
                    cb_mask.set_attribute("checked", "")
                else:
                    cb_mask.remove_attribute("checked")
            _set_text(doc, "cb-mask-label", tr("image_preview.show_mask_overlay"))
        else:
            if mask_sep:
                mask_sep.set_attribute("class", "separator hidden")
            if mask_label:
                mask_label.set_attribute("class", "sidebar-section hidden")
            if mask_name:
                mask_name.set_attribute("class", "sidebar-value hidden")
            if mask_row:
                mask_row.set_attribute("class", "sidebar-check-row hidden")

    def _update_status(self, doc, has_images: bool):
        info_el = doc.get_element_by_id("status-info")
        zoom_el = doc.get_element_by_id("status-zoom")

        if has_images:
            path = self._image_paths[self._current_index]
            w, h, _c = self._get_image_info(path)
            parts = []
            if w > 0 and h > 0:
                parts.append(f"{w}x{h}")
                mp = (w * h) / 1_000_000
                parts.append(f"{mp:.1f} MP")
            if path.exists():
                parts.append(self._format_size(path.stat().st_size))
            if info_el:
                info_el.set_inner_rml(_xml_escape(" \u00b7 ".join(parts)))
        else:
            if info_el:
                info_el.set_inner_rml("")

        zoom_text = f"{lf.ui.tr('image_preview.zoom')}: {self._get_zoom_display()}"
        if zoom_el:
            zoom_el.set_inner_rml(_xml_escape(zoom_text))

    # -- Keyboard --

    def _on_keydown(self, event):
        key = int(event.get_parameter("key_identifier", "0"))

        if key == KI_LEFT or key == KI_UP:
            self._navigate(-1)
            event.stop_propagation()
        elif key == KI_RIGHT or key == KI_DOWN:
            self._navigate(1)
            event.stop_propagation()
        elif key == KI_HOME:
            self._go_to_image(0)
            event.stop_propagation()
        elif key == KI_END:
            self._go_to_image(len(self._image_paths) - 1)
            event.stop_propagation()
        elif key == KI_F:
            self._fit_to_window = not self._fit_to_window
            if self._fit_to_window:
                self._zoom = 1.0
            self._dirty = True
            event.stop_propagation()
        elif key == KI_I:
            self._show_info = not self._show_info
            self._dirty = True
            event.stop_propagation()
        elif key == KI_T:
            self._show_filmstrip = not self._show_filmstrip
            if self._show_filmstrip:
                self._filmstrip_needs_rebuild = True
            self._dirty = True
            event.stop_propagation()
        elif key == KI_M:
            if self._has_valid_overlay():
                self._show_overlay = not self._show_overlay
                self._dirty = True
            event.stop_propagation()
        elif key == KI_1:
            self._zoom = 1.0
            self._fit_to_window = False
            self._dirty = True
            event.stop_propagation()
        elif key == KI_OEM_PLUS:
            self._zoom = min(ZOOM_MAX, self._zoom * 1.25)
            self._fit_to_window = False
            self._dirty = True
            event.stop_propagation()
        elif key == KI_OEM_MINUS:
            self._zoom = max(ZOOM_MIN, self._zoom / 1.25)
            self._fit_to_window = False
            self._dirty = True
            event.stop_propagation()
        elif key == KI_SPACE:
            if self._fit_to_window:
                self._fit_to_window = False
                self._zoom = 1.0
            else:
                self._fit_to_window = True
            self._dirty = True
            event.stop_propagation()
        elif key == KI_R:
            self._zoom = 1.0
            self._fit_to_window = True
            self._dirty = True
            event.stop_propagation()
        elif key == KI_ESCAPE:
            self._close_panel()
            event.stop_propagation()

    # -- Helpers --

    @staticmethod
    def _scroll_filmstrip(filmstrip, index: int):
        THUMB_H = 68  # 64dp height + 4dp margin
        item_top = index * THUMB_H
        item_bot = item_top + THUMB_H
        view_h = filmstrip.client_height
        if view_h <= 0:
            return
        st = filmstrip.scroll_top
        if item_top < st:
            filmstrip.scroll_top = item_top
        elif item_bot > st + view_h:
            filmstrip.scroll_top = item_bot - view_h

    def _get_zoom_display(self) -> str:
        if self._fit_to_window:
            return lf.ui.tr("image_preview.fit")
        return f"{self._zoom * 100:.0f}%"

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        if size_bytes >= 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.0f} KB"
        return f"{size_bytes} B"


def _set_text(doc, element_id: str, text: str):
    el = doc.get_element_by_id(element_id)
    if el:
        el.set_inner_rml(_xml_escape(text))


def open_image_preview(image_paths: list[Path], mask_paths: list[Path], start_index: int):
    if _instance:
        _instance.open(image_paths, mask_paths, start_index)
    lf.ui.set_panel_enabled("lfs.image_preview", True)


def open_camera_preview_by_uid(cam_uid: int):
    scene = lf.get_scene()
    if not scene:
        return
    target = None
    for node in scene.get_nodes():
        if node.type == lf.scene.NodeType.CAMERA and node.camera_uid == cam_uid:
            target = node
            break
    if not target or not target.image_path:
        return

    parent = scene.get_node_by_id(target.parent_id) if target.parent_id >= 0 else None
    child_ids = parent.children if parent else [n.id for n in scene.get_nodes() if n.type == lf.scene.NodeType.CAMERA]

    image_paths = []
    mask_paths = []
    start_index = 0
    for cid in child_ids:
        child = scene.get_node_by_id(cid)
        if not child or child.type != lf.scene.NodeType.CAMERA or not child.image_path:
            continue
        if child.id == target.id:
            start_index = len(image_paths)
        image_paths.append(Path(child.image_path))
        mask_paths.append(Path(child.mask_path) if child.mask_path else None)

    if image_paths:
        open_image_preview(image_paths, mask_paths, start_index)
