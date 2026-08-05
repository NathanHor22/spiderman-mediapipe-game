import unittest
from types import SimpleNamespace

from spidergame.render3d.buildings import (
    Bounds3D,
    VariantSpec,
    choose_variant,
    fit_variant_to_proxy,
    parse_manifest_data,
    proxy_key,
)


class BuildingRendererHelperTests(unittest.TestCase):
    def test_manifest_accepts_named_node_records(self):
        specs = parse_manifest_data(
            {
                "nodes": [
                    {
                        "node": "b1",
                        "bounds": {
                            "min": [-2, -3, 0],
                            "max": [2, 3, 12],
                        },
                    },
                    {
                        "node": "g1",
                        "bounds": [[-1, -1, 0], [1, 1, 10]],
                    },
                ]
            },
            require_expected=False,
        )

        self.assertEqual([spec.name for spec in specs], ["b1", "g1"])
        self.assertEqual(specs[0].bounds.size, (4.0, 6.0, 12.0))

    def test_manifest_converts_gltf_y_up_bounds_to_panda_axes(self):
        specs = parse_manifest_data(
            {
                "nodes": [
                    {
                        "name": "b1",
                        "local_bounds_y_up": {
                            "min": [-2, 0, -5],
                            "max": [3, 20, 7],
                        },
                    }
                ]
            },
            require_expected=False,
        )

        self.assertEqual(specs[0].bounds.minimum, (-2.0, -7.0, 0.0))
        self.assertEqual(specs[0].bounds.maximum, (3.0, 5.0, 20.0))

    def test_variant_choice_minimizes_aspect_distortion(self):
        squat = VariantSpec("b1", Bounds3D((0, 0, 0), (10, 10, 10)))
        tower = VariantSpec("b2", Bounds3D((0, 0, 0), (5, 5, 20)))

        selected = choose_variant(
            [squat, tower], (20, 20, 80), seed=9, key=(1, 2, 3)
        )

        self.assertEqual(selected.name, "b2")

    def test_variant_choice_is_independent_of_input_order(self):
        first = VariantSpec("b1", Bounds3D((0, 0, 0), (2, 2, 8)))
        second = VariantSpec("g1", Bounds3D((0, 0, 0), (2, 2, 8)))

        selected_a = choose_variant(
            [first, second], (20, 20, 80), seed=13, key=(4, 5, 6)
        )
        selected_b = choose_variant(
            [second, first], (20, 20, 80), seed=13, key=(4, 5, 6)
        )

        self.assertEqual(selected_a.name, selected_b.name)

    def test_fit_uses_world_x_and_z_as_panda_x_and_y(self):
        variant = VariantSpec(
            "b1",
            Bounds3D(minimum=(-2, -1, 1), maximum=(2, 3, 11)),
        )
        proxy = SimpleNamespace(
            x0=-34.0,
            x1=-14.0,
            z0=50.0,
            z1=90.0,
            height=100.0,
            side=-1,
        )

        fit = fit_variant_to_proxy(variant, proxy)

        self.assertEqual(fit.scale, (5.0, 10.0, 10.0))
        self.assertEqual(fit.position, (-24.0, 60.0, -10.0))
        mapped_min = tuple(
            pos + lo * scale
            for pos, lo, scale in zip(
                fit.position, variant.bounds.minimum, fit.scale
            )
        )
        mapped_max = tuple(
            pos + hi * scale
            for pos, hi, scale in zip(
                fit.position, variant.bounds.maximum, fit.scale
            )
        )
        self.assertEqual(mapped_min, (-34.0, 50.0, 0.0))
        self.assertEqual(mapped_max, (-14.0, 90.0, 100.0))

    def test_proxy_key_is_geometry_based(self):
        values = dict(x0=14, x1=34, z0=5, z1=30, height=80, side=1)
        self.assertEqual(
            proxy_key(SimpleNamespace(**values)),
            proxy_key(SimpleNamespace(**values)),
        )


if __name__ == "__main__":
    unittest.main()
