import unittest

from weeks import main


class TestParseCommandLineArguments(unittest.TestCase):
    def test_when_arg_not_set_then_returns_one_one(self):
        # Arrange
        expected = (1, 1)

        # Act
        actual = main.parse_command_line_arguments()

        # Assert
        self.assertEqual(actual, expected)


class TestMain(unittest.TestCase):
    def test_when_called_then_returns_zero(self):
        # Arrange
        expected = 0

        # Act
        actual = main.main()

        # Assert
        self.assertEqual(actual, expected)
