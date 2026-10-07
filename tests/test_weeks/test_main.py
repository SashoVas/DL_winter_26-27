import unittest

from weeks import main


class TestMain(unittest.TestCase):
    def test_when_called_then_returns_zero(self):
        # Arrange
        expected = 0

        # Act
        actual = main.main()

        # Assert
        self.assertEqual(actual, expected)
