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
        expected_array_len = "Array length: 100000"

        # Act
        task08.main()

        # Assert
        actual = self.capsys.readouterr().out
        self.assertIn("Loop sum:", actual)
        self.assertIn("Vectorized sum:", actual)
        self.assertIn("Vectorized speedup:", actual)
        self.assertIn("Vectorized mean:", actual)

        self.assertIn("Vectorized std:", actual)
        self.assertIn("Vectorized min/max:", actual)

    def test_when_ran_with_input_returns_results_for_the_current_input(
        self,
    ):
        # Arrange
        expected_array_len = "Array length: 100"

        # Act
        task08.main(100)

        # Assert
        actual = self.capsys.readouterr().out
        self.assertIn("Loop sum:", actual)
        self.assertIn("Vectorized sum:", actual)
        self.assertIn("Vectorized speedup:", actual)
        self.assertIn("Vectorized mean:", actual)

        self.assertIn("Vectorized std:", actual)
        self.assertIn("Vectorized min/max:", actual)
