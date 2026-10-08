import unittest

import pytest

from weeks.week01 import task14


class TestWeek01Task14(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_ran_returns_expected_output(self):
        # Arrange
        expected = """Original:     [2. 4. 4. 4. 5. 5. 7. 9.]
Min-max:      [0.         0.28571429 0.28571429 0.28571429 0.42857143 0.42857143
 0.71428571 1.        ]
Standardized: [-1.5 -0.5 -0.5 -0.5  0.   0.   1.   2. ]

Min-max range: [0.0000, 1.0000]
Standardized mean/std: 0.0000 / 1.0000"""

        # Act
        task14.main()

        # Assert
        actual = self.capsys.readouterr().out
        self.assertEqual(actual.strip(), expected)
