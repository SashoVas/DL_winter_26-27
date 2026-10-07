import importlib
import unittest


class TestRun(unittest.TestCase):
    def test_when_executed_then_no_errors_thrown(self):
        importlib.import_module("run")
