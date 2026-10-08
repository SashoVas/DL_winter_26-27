import unittest

import pytest

from weeks.week01 import task07


class TestWeek01Task07(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_ran_returns_expected_output(self):
        # Arrange
        expected = """Matrix:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

Second row: [5 6 7 8]
Third column: [ 3  7 11]
Submatrix:
[[2 3]
 [6 7]]

Boolean mask for more than 6:
[[False False False False]
 [False False  True  True]
 [ True  True  True  True]]
Values satisfying the mask: [ 7  8  9 10 11 12]"""

        # Act
        task07.main()

        # Assert
        actual = self.capsys.readouterr().out
        self.assertEqual(actual.strip(), expected)
