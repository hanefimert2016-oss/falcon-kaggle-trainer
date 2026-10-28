from __future__ import annotations

from types import SimpleNamespace

from factory_robot_3d.blender.animation import (
    _animation_fcurves,
    _set_linear_interpolation,
)


def _curve(*values: str):
    return SimpleNamespace(
        keyframe_points=[SimpleNamespace(interpolation=value) for value in values]
    )


def test_animation_fcurves_supports_legacy_action_shape():
    curve = _curve("BEZIER")
    animated = SimpleNamespace(
        animation_data=SimpleNamespace(
            action=SimpleNamespace(fcurves=(curve,))
        )
    )

    assert _animation_fcurves(animated) == (curve,)


def test_animation_fcurves_supports_blender5_layered_channelbags():
    curve_a = _curve("BEZIER")
    curve_b = _curve("CONSTANT")
    bag = SimpleNamespace(fcurves=(curve_a, curve_b))
    strip = SimpleNamespace(channelbags=(bag,))
    layer = SimpleNamespace(strips=(strip,))
    animated = SimpleNamespace(
        animation_data=SimpleNamespace(
            action=SimpleNamespace(layers=(layer,))
        )
    )

    assert _animation_fcurves(animated) == (curve_a, curve_b)


def test_linear_interpolation_updates_layered_action_keyframes():
    curve = _curve("BEZIER", "CONSTANT", "SINE")
    bag = SimpleNamespace(fcurves=(curve,))
    strip = SimpleNamespace(channelbags=(bag,))
    layer = SimpleNamespace(strips=(strip,))
    animated = SimpleNamespace(
        animation_data=SimpleNamespace(
            action=SimpleNamespace(layers=(layer,))
        )
    )

    _set_linear_interpolation(animated)

    assert [point.interpolation for point in curve.keyframe_points] == [
        "LINEAR",
        "LINEAR",
        "LINEAR",
    ]


def test_camera_bezier_interpolation_supports_blender5_layered_actions():
    from factory_robot_3d.blender.cameras import _set_camera_bezier_interpolation

    points = [
        SimpleNamespace(
            interpolation="LINEAR",
            handle_left_type="FREE",
            handle_right_type="FREE",
        ),
        SimpleNamespace(
            interpolation="CONSTANT",
            handle_left_type="ALIGNED",
            handle_right_type="ALIGNED",
        ),
    ]
    curve = SimpleNamespace(keyframe_points=points)
    bag = SimpleNamespace(fcurves=(curve,))
    strip = SimpleNamespace(channelbags=(bag,))
    layer = SimpleNamespace(strips=(strip,))
    camera = SimpleNamespace(
        animation_data=SimpleNamespace(
            action=SimpleNamespace(layers=(layer,))
        )
    )

    _set_camera_bezier_interpolation(camera)

    assert [p.interpolation for p in points] == ["BEZIER", "BEZIER"]
    assert [p.handle_left_type for p in points] == ["AUTO_CLAMPED", "AUTO_CLAMPED"]
    assert [p.handle_right_type for p in points] == ["AUTO_CLAMPED", "AUTO_CLAMPED"]
