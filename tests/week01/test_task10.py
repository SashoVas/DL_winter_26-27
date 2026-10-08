import unittest

import pytest

from weeks.week01 import task10


class TestWeek01Task10(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_ran_returns_expected_output(self):
        # Arrange
        expected = """A:
[[1 2]
 [3 4]]

B:
[[5 6]
 [7 8]]

Elementwise:
[[ 5 12]
 [21 32]]

Matrix multiplication:
[[19 22]
 [43 50]]

The transpose of A:
[[1 3]
 [2 4]]"""

        # Act
        task10.main()

        # Assert
        actual = self.capsys.readouterr().out
        self.assertEqual(actual.strip(), expected)
