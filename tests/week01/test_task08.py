import unittest

import pytest

from weeks.week01 import task08


class TestWeek01Task08(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_ran_with_no_input_returns_results_output_for_vectro_with_lenght_100000(
        self,
    ):
        # Arrange
        expected_array_len = "Array length: 100"

        expected_outputs = [
            "Loop sum:",
            "Vectorized sum:",
            "Vectorized speedup:",
            "Vectorized mean:",
            "Vectorized std:",
            "Vectorized min/max:",
        ]

        # Act
        task08.main()

        # Assert
        actual = self.capsys.readouterr().out
        for i in expected_outputs:
            self.assertIn(i, actual)

        self.assertIn(expected_array_len, actual)

    def test_when_ran_with_input_returns_results_for_the_current_input(
        self,
    ):
        # Arrange
        expected_array_len = "Array length: 100"
        expected_outputs = [
            "Loop sum:",
            "Vectorized sum:",
            "Vectorized speedup:",
            "Vectorized mean:",
            "Vectorized std:",
            "Vectorized min/max:",
        ]
        # Act
        task08.main(100)

        # Assert
        actual = self.capsys.readouterr().out
        for i in expected_outputs:
            self.assertIn(i, actual)

        self.assertIn(expected_array_len, actual)
