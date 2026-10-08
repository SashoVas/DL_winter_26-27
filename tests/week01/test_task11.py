import unittest
from unittest.mock import patch

import pytest

from weeks.week01 import task11


class TestWeek01Task11(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    @patch("weeks.week01.task11.plt.show")
    def test_when_ran_with_no_input_returns_results_output_for_default_params(
        self, mock_show
    ):
        # Arrange
        expected_equation = "Generated 50 points for y = 2*x + 1 + noise"

        # Act
        task11.main()

        # Assert
        actual = self.capsys.readouterr().out
        self.assertIn(expected_equation, actual)
        self.assertIn("x range: [0, 10]", actual)
        self.assertIn("y range: ", actual)
        mock_show.assert_called_once()

    @patch("weeks.week01.task11.plt.show")
    def test_when_ran_with_correct_input_then_prints_correct_outputs(self, mock_show):
        # Arrange
        num_points = 30
        slope = 3.2
        intercept = 4.4
        range = (-2, 9)
        expected_equation = (
            f"Generated {num_points} points for y = {slope}*x + {intercept} + noise"
        )

        # Act
        task11.main(
            slope,
            intercept,
            num_points,
            range,
        )

        # Assert
        actual = self.capsys.readouterr().out
        self.assertIn(expected_equation, actual)
        self.assertIn(f"x range: {list(range)}", actual)
        self.assertIn("y range: ", actual)
        mock_show.assert_called_once()
