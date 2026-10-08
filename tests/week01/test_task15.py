import unittest
import pytest
from weeks.week01 import task15


class TestWeek01Task15(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_ran_returns_expected_output(self):
        # Arrange
        expected = """NumPy array:
[[1 2 3]
 [4 5 6]]
dtype=int64, shape=(2, 3)

PyTorch tensor:
tensor([[1, 2, 3],
        [4, 5, 6]])
dtype=torch.int64, shape=torch.Size([2, 3]), device=cpu

Selected device: cpu
Tensor moved to device: cpu, dtype=torch.float32"""

        # Act
        task15.main()

        # Assert
        actual = self.capsys.readouterr().out
        self.assertEqual(actual.strip(), expected)
