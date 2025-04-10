import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from brainglobe_utils.cells.cells import Cell

from cellfinder.core.classify.classify import (
    ClassificationParameters,
    DataParameters,
    classify_with_params,
    main,
)


class TestClassificationParameters(unittest.TestCase):
    def test_initialization_with_defaults(self):
        """Test that ClassificationParameters initializes
        with default values."""
        params = ClassificationParameters()
        self.assertEqual(params.batch_size, 64)
        self.assertEqual(params.cube_height, 50)
        self.assertEqual(params.cube_width, 50)
        self.assertEqual(params.cube_depth, 20)
        self.assertEqual(params.network_depth, "50-layer")
        self.assertEqual(params.max_workers, 3)

    def test_initialization_with_custom_values(self):
        """Test that ClassificationParameters initializes
        with custom values."""
        params = ClassificationParameters(
            batch_size=32,
            cube_height=40,
            cube_width=40,
            cube_depth=15,
            network_depth="34-layer",
            max_workers=2,
        )
        self.assertEqual(params.batch_size, 32)
        self.assertEqual(params.cube_height, 40)
        self.assertEqual(params.cube_width, 40)
        self.assertEqual(params.cube_depth, 15)
        self.assertEqual(params.network_depth, "34-layer")
        self.assertEqual(params.max_workers, 2)


class TestDataParameters(unittest.TestCase):
    def test_initialization_with_defaults(self):
        """Test that DataParameters initializes with default free CPU value."""
        params = DataParameters(
            voxel_sizes=(5, 2, 2), network_voxel_sizes=(5, 2, 2)
        )
        self.assertEqual(params.voxel_sizes, (5, 2, 2))
        self.assertEqual(params.network_voxel_sizes, (5, 2, 2))
        self.assertEqual(params.n_free_cpus, 2)

    def test_initialization_with_custom_values(self):
        """Test that DataParameters initializes with custom values."""
        params = DataParameters(
            voxel_sizes=(10, 5, 5),
            network_voxel_sizes=(5, 2, 2),
            n_free_cpus=4,
        )
        self.assertEqual(params.voxel_sizes, (10, 5, 5))
        self.assertEqual(params.network_voxel_sizes, (5, 2, 2))
        self.assertEqual(params.n_free_cpus, 4)


class TestClassifyWithParams(unittest.TestCase):
    def setUp(self):
        """Set up common test data."""
        self.signal_array = np.zeros((10, 100, 100), dtype=np.float32)
        self.background_array = np.zeros((10, 100, 100), dtype=np.float32)
        self.points = [Cell([50, 50, 5], Cell.CELL)]
        self.data_params = DataParameters(
            voxel_sizes=(5, 2, 2), network_voxel_sizes=(5, 2, 2)
        )
        self.classification_params = ClassificationParameters()

    @patch("cellfinder.core.classify.classify.CubeGeneratorFromFile")
    @patch("cellfinder.core.classify.classify.get_model")
    @patch("cellfinder.core.classify.classify.get_num_processes")
    def test_basic_functionality(
        self, mock_get_num_processes, mock_get_model, mock_cube_generator
    ):
        """Test that classify_with_params calls the correct functions
        with right parameters."""
        mock_get_num_processes.return_value = 2
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_model.predict.return_value = np.array([[0.9, 0.1]])

        mock_generator_instance = MagicMock()
        mock_cube_generator.return_value = mock_generator_instance
        mock_generator_instance.ordered_points = self.points

        result = classify_with_params(
            points=self.points,
            signal_array=self.signal_array,
            background_array=self.background_array,
            data_parameters=self.data_params,
            classification_parameters=self.classification_params,
        )

        mock_get_num_processes.assert_called_once_with(min_free_cpu_cores=2)
        mock_cube_generator.assert_called_once()
        mock_get_model.assert_called_once()
        mock_model.predict.assert_called_once()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].type, 1)

    @patch("cellfinder.core.classify.classify.CubeGeneratorFromFile")
    @patch("cellfinder.core.classify.classify.get_model")
    @patch("cellfinder.core.classify.classify.get_num_processes")
    @patch("cellfinder.core.tools.prep.prep_model_weights")
    def test_auto_model_weights(
        self,
        mock_prep_model_weights,
        mock_get_num_processes,
        mock_get_model,
        mock_cube_generator,
    ):
        """Test that model weights are automatically prepared
        when not provided."""
        mock_get_num_processes.return_value = 2
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_model.predict.return_value = np.array([[0.9, 0.1]])
        mock_prep_model_weights.return_value = "mock_weights_path"

        mock_generator_instance = MagicMock()
        mock_cube_generator.return_value = mock_generator_instance
        mock_generator_instance.ordered_points = self.points

        classify_with_params(
            points=self.points,
            signal_array=self.signal_array,
            background_array=self.background_array,
            data_parameters=self.data_params,
            classification_parameters=self.classification_params,
            model_name="resnet50_tv",
        )

        mock_prep_model_weights.assert_called_once_with(
            None, None, "resnet50_tv"
        )

        mock_get_model.assert_called_once_with(
            existing_model=None,
            model_weights="mock_weights_path",
            network_depth="50-layer",
            inference=True,
        )


class TestMainBackwardCompatibility(unittest.TestCase):
    def setUp(self):
        """Set up common test data."""
        self.signal_array = np.zeros((10, 100, 100), dtype=np.float32)
        self.background_array = np.zeros((10, 100, 100), dtype=np.float32)
        self.points = [Cell([50, 50, 5], Cell.CELL)]
        self.voxel_sizes = (5, 2, 2)
        self.network_voxel_sizes = (5, 2, 2)

    @patch("cellfinder.core.classify.classify.classify_with_params")
    def test_main_wrapper_calls_classify_with_params(
        self, mock_classify_with_params
    ):
        """Test that the main function correctly calls classify_with_params."""
        mock_classify_with_params.return_value = self.points

        result = main(
            points=self.points,
            signal_array=self.signal_array,
            background_array=self.background_array,
            n_free_cpus=2,
            voxel_sizes=self.voxel_sizes,
            network_voxel_sizes=self.network_voxel_sizes,
            batch_size=32,
            cube_height=40,
            cube_width=40,
            cube_depth=15,
            trained_model=None,
            model_weights=None,
            network_depth="34",
            max_workers=4,
        )

        self.assertEqual(result, self.points)

        mock_classify_with_params.assert_called_once()

        args, kwargs = mock_classify_with_params.call_args

        self.assertEqual(args[0], self.points)
        self.assertTrue(np.array_equal(args[1], self.signal_array))
        self.assertTrue(np.array_equal(args[2], self.background_array))

        data_params = args[3]
        self.assertEqual(data_params.voxel_sizes, self.voxel_sizes)
        self.assertEqual(
            data_params.network_voxel_sizes, self.network_voxel_sizes
        )
        self.assertEqual(data_params.n_free_cpus, 2)

        classification_params = args[4]
        self.assertEqual(classification_params.batch_size, 32)
        self.assertEqual(classification_params.cube_height, 40)
        self.assertEqual(classification_params.cube_width, 40)
        self.assertEqual(classification_params.cube_depth, 15)
        self.assertEqual(classification_params.network_depth, "34-layer")
        self.assertEqual(classification_params.max_workers, 4)

        self.assertEqual(args[5], None)
        self.assertEqual(args[6], None)


if __name__ == "__main__":
    unittest.main()
