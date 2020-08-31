import numpy as np
import pytest

from umapviz import umap_metrics as um


class DescribeMetrics:
    @pytest.mark.parametrize(
        "a_cat, b_cat, weights, expected_metric",
        [
            (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), np.array([1, 1, 1, 1]), 0),
            (np.array([1, 2, 3, 4]), np.array([2, 2, 3, 4]), np.array([1, 1, 1, 1]), 1),
            (np.array([1, 2, 3, 4]), np.array([2, 1, 3, 4]), np.array([1, 1, 1, 1]), 2),
            (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), np.array([2, 1, 1, 1]), 0),
            (np.array([1, 2, 3, 4]), np.array([2, 2, 3, 4]), np.array([2, 1, 1, 1]), 2),
            (np.array([1, 2, 3, 4]), np.array([3, 2, 3, 4]), np.array([2, 1, 1, 1]), 2),
            (
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 8]),
                np.array([0.5, 0.5, 1, 1]),
                1.5,
            ),
        ],
    )
    def test_gower_dist_categorical(self, a_cat, b_cat, weights, expected_metric):
        metric = um.gower_dist_categorical(a_cat, b_cat, weights)

        assert type(metric) == float
        assert metric >= 0
        assert metric == expected_metric

    @pytest.mark.parametrize(
        "a_num, b_num, weights, expected_metric",
        [
            (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), np.array([1, 1, 1, 1]), 0),
            (np.array([1, 2, 3, 4]), np.array([2, 2, 3, 4]), np.array([1, 1, 1, 1]), 1),
            (np.array([1, 2, 3, 4]), np.array([2, 1, 3, 4]), np.array([1, 1, 1, 1]), 2),
            (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), np.array([2, 1, 1, 1]), 0),
            (np.array([1, 2, 3, 4]), np.array([2, 2, 3, 4]), np.array([2, 1, 1, 1]), 2),
            (np.array([1, 2, 3, 4]), np.array([3, 2, 3, 4]), np.array([2, 1, 1, 1]), 2),
            (
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 8]),
                np.array([0.5, 0.5, 1, 1]),
                1.5,
            ),
        ],
    )
    def test_gower_dist_numerical_old(self, a_num, b_num, weights, expected_metric):
        metric = um.gower_dist_numerical_old(a_num, b_num, weights)

        assert type(metric) == float
        assert metric >= 0
        assert metric == expected_metric

    @pytest.mark.parametrize(
        "a_num, b_num, weights, min_vals, max_vals, expected_metric",
        [
            (
                np.array([1, 2, 3, 4]),
                np.array([1, 2, 3, 4]),
                np.array([1, 1, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2, 3, 4]),
                0,
            ),
            (
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 4]),
                np.array([1, 1, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 4]),
                4,
            ),
            (
                np.array([1, 2, 3, 4]),
                np.array([2, 1, 3, 4]),
                np.array([1, 1, 1, 1]),
                np.array([1, 1, 3, 4]),
                np.array([1, 1, 3, 4]),
                2,
            ),
            (
                np.array([1, 2, 3, 4]),
                np.array([1, 2, 3, 4]),
                np.array([2, 1, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2, 3, 4]),
                0,
            ),
            (
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 4]),
                np.array([2, 1, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2, 3, 4]),
                2,
            ),
            (
                np.array([1, 2, 3, 4]),
                np.array([3, 2, 3, 4]),
                np.array([2, 1, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([3, 2, 3, 4]),
                5,
            ),
            (
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 8]),
                np.array([0.5, 0.5, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 4]),
                3,
            ),
        ],
    )
    def test_gower_dist_numerical(
        self, a_num, b_num, weights, min_vals, max_vals, expected_metric
    ):
        metric = um.gower_dist_numerical(a_num, b_num, weights, min_vals, max_vals)

        assert type(metric) == float
        assert metric >= 0
        assert metric == expected_metric
