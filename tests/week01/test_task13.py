import unittest
import pytest
from weeks.week01 import task13
from unittest.mock import patch


class TestWeek01Task13(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    @patch("weeks.week01.task13.plt.show")
    def test_when_ran_with_no_input_then_returns_results_output_for_default_params(
        self, mock_show
    ):
        # Arrange
        expected_histogram_messages = ["Uniform:", "Normal:", "Exponential:"]

        # Act
        task13.main()

        # Assert
        actual = self.capsys.readouterr().out
        for i in expected_histogram_messages:
            self.assertIn(i, actual)
        mock_show.assert_called_once()
