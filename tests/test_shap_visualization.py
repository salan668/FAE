import unittest

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import pandas as pd

from BC.Visualization.FeatureSort import SHAPBeeswarmPlot


class ShapVisualizationTest(unittest.TestCase):
    def test_beeswarm_colors_dots_by_feature_value(self):
        """The colorbar must communicate feature value, not duplicate SHAP position."""
        shap_df = pd.DataFrame({'a': [-0.4, 0.3], 'b': [0.2, -0.1]})
        feature_df = pd.DataFrame({'a': [1.0, 9.0], 'b': [2.0, 8.0]})

        axis = SHAPBeeswarmPlot(shap_df, feature_df, fig=Figure())

        self.assertEqual(axis.get_figure().axes[-1].get_ylabel(), 'Feature value')


if __name__ == '__main__':
    unittest.main()
