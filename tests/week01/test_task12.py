import unittest
from unittest.mock import patch

import pytest

from weeks.week01 import task12


class TestWeek01Task12(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    @patch("weeks.week01.task12.plt.show")
    def test_when_ran_with_no_input_then_returns_results_output_for_default_params(
        self, mock_show
    ):
        # Arrange
        expected_histogram_message = "Histogram sample count: 1000"

        # Act
        task12.main()

        # Assert
        actual = self.capsys.readouterr().out
        self.assertIn("sin(x) range:", actual)
        self.assertIn(expected_histogram_message, actual)
        mock_show.assert_called_once()

    @patch("weeks.week01.task12.plt.show")
    def test_when_ran_with_correct_input_then_prints_correct_outputs(self, mock_show):
        # Arrange
        samples = 10
        expected_histogram_message = f"Histogram sample count: {samples}"

        # Act
        task12.main(samples)

        # Assert
        actual = self.capsys.readouterr().out
        self.assertIn("sin(x) range:", actual)
        self.assertIn(expected_histogram_message, actual)
        mock_show.assert_called_once()
