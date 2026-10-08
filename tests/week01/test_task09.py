import unittest
import pytest
from weeks.week01 import task09


class TestWeek01Task09(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_ran_returns_expected_output(self):
        # Arrange
        expected = """Matrix + scalar (10):
[[11 12 13]
 [14 15 16]]

Matrix (2, 3) + row vector (3,):
[[11 22 33]
 [14 25 36]]

Matrix (2, 3) + column vector (2, 1):
[[101 102 103]
 [204 205 206]]

Matrix (2, 3) + incompatible (2,) raises:
operands could not be broadcast together with shapes (2,3) (2,)"""

        # Act
        task09.main()

        # Assert
        actual = self.capsys.readouterr().out
        self.assertEqual(actual.strip(), expected)
