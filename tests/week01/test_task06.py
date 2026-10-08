import unittest

import pytest

from weeks.week01 import task06


class TestWeek01Task06(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_ran_returns_expected_output(self):
        # Arrange
        expected = """1D array (vector): shape=(5,), dtype=int64, ndim=1, size=5
[1 2 3 4 5]

2D array (matrix): shape=(2, 3), dtype=int64, ndim=2, size=6
[[1 2 3]
 [4 5 6]]

3D array (volume): shape=(2, 3, 4), dtype=int64, ndim=3, size=24
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]"""

        # Act
        task06.main()

        # Assert
        actual = self.capsys.readouterr().out
        self.assertEqual(actual.strip(), expected)
