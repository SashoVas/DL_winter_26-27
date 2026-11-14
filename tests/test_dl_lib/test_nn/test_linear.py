import unittest

import torch

from dl_lib.nn.linear import Linear


class TestLinearInit(unittest.TestCase):
    def test_when_constructed_then_weight_and_bias_have_torch_shapes(self):
        # Arrange
        in_features, out_features = 256, 128
        expected_weight_shape = (out_features, in_features)
        expected_bias_shape = (out_features,)

        # Act
        linear = Linear(in_features, out_features)

        # Assert
        self.assertEqual(tuple(linear.weight.shape), expected_weight_shape)
        self.assertEqual(tuple(linear.bias.shape), expected_bias_shape)

    def test_when_constructed_then_weight_and_bias_require_grad(self):
        # Arrange & Act
        linear = Linear(in_features=10, out_features=5)

        # Assert
        self.assertTrue(linear.weight.requires_grad)
        self.assertTrue(linear.bias.requires_grad)


class TestLinearForward(unittest.TestCase):
    def test_when_input_is_unbatched_then_output_is_unbatched(self):
        # Arrange
        in_features, out_features = 10, 5
        linear = Linear(in_features, out_features)
        x = torch.randn(in_features)

        # Act
        actual = linear(x)

        # Assert
        self.assertEqual(actual.dim(), 1)
        self.assertEqual(tuple(actual.shape), (out_features,))

    def test_when_input_is_batched_then_output_keeps_the_batch_dimension(self):
        # Arrange
        batch_size, in_features, out_features = 4, 10, 5
        linear = Linear(in_features, out_features)
        x = torch.randn(batch_size, in_features)

        # Act
        actual = linear(x)

        # Assert
        self.assertEqual(actual.shape[0], batch_size)
        self.assertEqual(actual.shape[1], out_features)

    def test_when_input_has_multiple_batch_dimensions_then_preserves_them(self):
        # Arrange
        in_features, out_features = 10, 5
        linear = Linear(in_features, out_features)
        x = torch.randn(2, 3, in_features)
        expected_shape = (2, 3, out_features)

        # Act
        actual = linear(x)

        # Assert
        self.assertEqual(tuple(actual.shape), expected_shape)

    def test_when_given_same_weights_then_matches_torch_nn_linear(self):
        # Arrange
        torch.manual_seed(0)
        in_features, out_features = 10, 5
        x = torch.randn(4, in_features)

        ours = Linear(in_features, out_features)
        reference = torch.nn.Linear(in_features, out_features)

        with torch.no_grad():
            ours.weight.copy_(reference.weight)
            ours.bias.copy_(reference.bias)

        # Act
        actual = ours(x)
        expected = reference(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_when_unbatched_then_matches_torch_nn_linear(self):
        # Arrange
        torch.manual_seed(1)
        in_features, out_features = 10, 5
        x = torch.randn(in_features)

        ours = Linear(in_features, out_features)
        reference = torch.nn.Linear(in_features, out_features)

        with torch.no_grad():
            ours.weight.copy_(reference.weight)
            ours.bias.copy_(reference.bias)

        # Act
        actual = ours(x)
        expected = reference(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_when_input_has_multiple_batch_dimensions_then_matches_torch_nn_linear(
        self,
    ):
        # Arrange
        torch.manual_seed(2)
        in_features, out_features = 16, 8
        x = torch.randn(2, 3, 4, in_features)

        ours = Linear(in_features, out_features)
        reference = torch.nn.Linear(in_features, out_features)

        with torch.no_grad():
            ours.weight.copy_(reference.weight)
            ours.bias.copy_(reference.bias)

        # Act
        actual = ours(x)
        expected = reference(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_when_called_then_gradients_are_computed(self):
        # Arrange
        in_features, out_features = 10, 5
        linear = Linear(in_features, out_features)
        x = torch.randn(4, in_features, requires_grad=True)

        # Act
        actual = linear(x)
        actual.sum().backward()

        # Assert
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(linear.weight.grad)
        self.assertIsNotNone(linear.bias.grad)


class TestLinearParameters(unittest.TestCase):
    def test_when_called_then_returns_weight_and_bias(self):
        # Arrange
        linear = Linear(in_features=10, out_features=5)
        expected = [linear.weight, linear.bias]

        # Act
        actual = linear.parameters()

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_then_returned_parameters_are_the_same_objects(self):
        # Arrange
        linear = Linear(in_features=10, out_features=5)

        # Act
        actual = linear.parameters()

        # Assert
        self.assertIs(actual[0], linear.weight)
        self.assertIs(actual[1], linear.bias)


class TestLinearTo(unittest.TestCase):
    def test_when_called_then_returns_self(self):
        # Arrange
        linear = Linear(in_features=10, out_features=5)

        # Act
        actual = linear.to("cpu")

        # Assert
        self.assertIs(actual, linear)

    def test_when_called_then_moves_parameters_to_device(self):
        # Arrange
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        linear = Linear(in_features=10, out_features=5)

        # Act
        linear.to(device)

        # Assert
        self.assertEqual(linear.weight.device, device)
        self.assertEqual(linear.bias.device, device)

    def test_when_called_then_preserves_parameter_values(self):
        # Arrange
        linear = Linear(in_features=10, out_features=5)
        expected_weight = linear.weight.detach().clone()
        expected_bias = linear.bias.detach().clone()

        # Act
        linear.to("cpu")

        # Assert
        self.assertTrue(torch.equal(linear.weight, expected_weight))
        self.assertTrue(torch.equal(linear.bias, expected_bias))

    def test_when_called_then_parameters_still_require_grad(self):
        # Arrange
        linear = Linear(in_features=10, out_features=5)

        # Act
        linear.to("cpu")

        # Assert
        self.assertTrue(linear.weight.requires_grad)
        self.assertTrue(linear.bias.requires_grad)

    def test_when_called_then_parameters_are_leaf_tensors(self):
        # Arrange
        linear = Linear(in_features=10, out_features=5)

        # Act
        linear.to("cpu")

        # Assert
        self.assertTrue(linear.weight.is_leaf)
        self.assertTrue(linear.bias.is_leaf)
