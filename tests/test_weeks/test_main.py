import sys
import unittest

from weeks import main


class TestParseCommandLineArguments(unittest.TestCase):
    def setUp(self) -> None:
        self.original_argv = sys.argv

    def tearDown(self) -> None:
        sys.argv = self.original_argv

    def test_when_arg_not_set_then_returns_one_one(self):
        # Arrange
        expected = (1, 1)
        sys.argv = [self.original_argv[0]]

        # Act
        actual = main.parse_command_line_arguments()

        # Assert
        self.assertEqual(actual, expected)

    def test_when_only_week_set_then_returns_that_week_and_one(self):
        # Arrange
        expected_week = 5
        expected = (expected_week, 1)
        sys.argv = [self.original_argv[0], "-w", str(expected_week)]

        # Act
        actual = main.parse_command_line_arguments()

        # Assert
        self.assertEqual(actual, expected)

    def test_when_only_task_set_then_returns_that_task_and_one(self):
        # Arrange
        expected_task = 12
        expected = (1, expected_task)
        sys.argv = [self.original_argv[0], "-t", str(expected_task)]

        # Act
        actual = main.parse_command_line_arguments()

        # Assert
        self.assertEqual(actual, expected)

    def test_when_task_and_week_set_then_returns_that_task_and_that_week(self):
        # Arrange
        expected_task = 12
        expected_week = 2
        expected = (expected_week, expected_task)
        sys.argv = [
            self.original_argv[0],
            "--week",
            str(expected_week),
            "--task",
            str(expected_task),
        ]

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
