import unittest
from unittest.mock import patch

import numpy as np
from brainglobe_utils.cells.cells import Cell

from cellfinder.core.classify import ClassificationParameters, DataParameters
from cellfinder.core.main import main as cellfinder_run


class TestClassificationIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test data for integration tests."""
        # Create small 3D arrays for testing
        self.signal_array = np.zeros((10, 100, 100), dtype=np.float32)
        self.background_array = np.zeros((10, 100, 100), dtype=np.float32)
        self.voxel_sizes = (5, 2, 2)

        # Mock detected cells
        self.detected_cells = [
            Cell([50, 50, 5], Cell.CELL),
            Cell([30, 40, 3], Cell.CELL),
        ]

    @patch("cellfinder.core.detect.main")
    @patch("cellfinder.core.classify.classify")
    @patch("cellfinder.core.tools.prep.prep_model_weights")
    def test_main_workflow_calls_classify_correctly(
        self, mock_prep_model_weights, mock_classify, mock_detect
    ):
        """Test that the main workflow correctly calls the
        classification module."""
        mock_detect.return_value = self.detected_cells
        mock_prep_model_weights.return_value = "mock_weights_path"
        mock_classify.return_value = self.detected_cells

        result = cellfinder_run(
            signal_array=self.signal_array,
            background_array=self.background_array,
            voxel_sizes=self.voxel_sizes,
            batch_size=32,
            network_depth="34",
        )

        self.assertEqual(result, self.detected_cells)

        mock_classify.assert_called_once()

        args, kwargs = mock_classify.call_args

        self.assertEqual(args[0], self.detected_cells)

        self.assertIsInstance(kwargs["data_parameters"], DataParameters)
        self.assertEqual(
            kwargs["data_parameters"].voxel_sizes, self.voxel_sizes
        )

        self.assertIsInstance(
            kwargs["classification_parameters"], ClassificationParameters
        )
        self.assertEqual(kwargs["classification_parameters"].batch_size, 32)
        self.assertEqual(
            kwargs["classification_parameters"].network_depth, "34-layer"
        )

    @patch("cellfinder.core.detect.main")
    def test_empty_detection_skips_classification(self, mock_detect):
        """Test that the main workflow skips classification when
        detection returns no cells."""
        mock_detect.return_value = []

        result = cellfinder_run(
            signal_array=self.signal_array,
            background_array=self.background_array,
            voxel_sizes=self.voxel_sizes,
        )

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
