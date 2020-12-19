import numpy as np
import pytest

from umapviz import umap_metrics as um


class DescribeMetrics:
    @pytest.mark.parametrize(
        "a_cat, b_cat, weights, expected_metric",
        [  # equal arrays
            (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), np.array([1, 1, 1, 1]), 0),
            # one element different (absolute difference of 1)
            (np.array([1, 2, 3, 4]), np.array([2, 2, 3, 4]), np.array([1, 1, 1, 1]), 1),
            # two elements different (absolute difference of 1 for each)
            (np.array([1, 2, 3, 4]), np.array([2, 1, 3, 4]), np.array([1, 1, 1, 1]), 2),
            # equal arrays - double weight for the first element
            (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), np.array([2, 1, 1, 1]), 0),
            # one element different (absolute difference of 1) but double weight
            (np.array([1, 2, 3, 4]), np.array([2, 2, 3, 4]), np.array([2, 1, 1, 1]), 2),
            # one element different (absolute difference of 2) but double weight
            (np.array([1, 2, 3, 4]), np.array([3, 2, 3, 4]), np.array([2, 1, 1, 1]), 2),
            # two elements different (absolute difference of 1 and 4 respectively)
            # float weights
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
        [  # equal arrays
            (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), np.array([1, 1, 1, 1]), 0),
            # one element different (absolute difference of 1)
            (np.array([1, 2, 3, 4]), np.array([2, 2, 3, 4]), np.array([1, 1, 1, 1]), 1),
            # two elements different (absolute difference of 1 for each)
            (np.array([1, 2, 3, 4]), np.array([2, 1, 3, 4]), np.array([1, 1, 1, 1]), 2),
            # equal arrays - double weight for the first element
            (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), np.array([2, 1, 1, 1]), 0),
            # one element different (absolute difference of 1) but double weight
            (np.array([1, 2, 3, 4]), np.array([2, 2, 3, 4]), np.array([2, 1, 1, 1]), 2),
            # one element different (absolute difference of 2) but double weight
            (np.array([1, 2, 3, 4]), np.array([3, 2, 3, 4]), np.array([2, 1, 1, 1]), 2),
            # two elements different (absolute difference of 1 and 4 respectively)
            # float weights
            (
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 8]),
                np.array([0.5, 0.5, 1, 1]),
                1.5,
            ),
            # points/elements having one component with much higher value than the other
            # components
            (
                np.array([100, 2, 3, 4]),
                np.array([2, 2, 3, 4]),
                np.array([1, 1, 1, 1]),
                1,
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
        [  # equal arrays
            (
                np.array([1, 2, 3, 4]),
                np.array([1, 2, 3, 4]),
                np.array([1, 1, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2, 3, 4]),
                0,
            ),
            # one element different (absolute difference of 1)
            (
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 4]),
                np.array([1, 1, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 4]),
                4,
            ),
            # two elements different (absolute difference of 1 for each)
            (
                np.array([1, 2, 3, 4]),
                np.array([2, 1, 3, 4]),
                np.array([1, 1, 1, 1]),
                np.array([1, 1, 3, 4]),
                np.array([1, 1, 3, 4]),
                2,
            ),
            # equal arrays - double weight for the first element
            (
                np.array([1, 2, 3, 4]),
                np.array([1, 2, 3, 4]),
                np.array([2, 1, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2, 3, 4]),
                0,
            ),
            # one element different (absolute difference of 1) but double weight
            (
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 4]),
                np.array([2, 1, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2, 3, 4]),
                2,
            ),
            # one element different (absolute difference of 2) but double weight
            (
                np.array([1, 2, 3, 4]),
                np.array([3, 2, 3, 4]),
                np.array([2, 1, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([3, 2, 3, 4]),
                5,
            ),
            # two elements different (absolute difference of 1 and 4 respectively)
            # float weights
            (
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 8]),
                np.array([0.5, 0.5, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 4]),
                3,
            ),
            # One point with out of scale value
            (
                np.array([1, 2, 3, 4]),
                np.array([100, 2, 3, 4]),
                np.array([2, 1, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 4]),
                495.0,
            ),
            # One point with high value, close to the upper bound of the value range
            (
                np.array([1, 2, 3, 4]),
                np.array([100, 2, 3, 4]),
                np.array([2, 1, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([100, 2, 3, 4]),
                5.0,
            ),
            # One point with low first component value and a high value of the value
            # range upper bound
            (
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 4]),
                np.array([2, 1, 1, 1]),
                np.array([1, 2, 3, 4]),
                np.array([100, 2, 3, 4]),
                0.05050505050505051,
            ),
            # Same as previous case but with switched `min_vals` and `max_vals`
            (
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 3, 4]),
                np.array([2, 1, 1, 1]),
                np.array([100, 2, 3, 4]),
                np.array([1, 2, 3, 4]),
                0.050505050505050275,
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
