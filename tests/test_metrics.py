import unittest
import pandas as pd
from utils.metrics import Metrics

class TestMetrics(unittest.TestCase):

    def test_recall(self):
        m = Metrics()

        # Perfect recall
        pred = [("car", "inFrontOf", "ego")]
        target = [("car", "inFrontOf", "ego")]
        recall = m.calculate_recall(pred, target)
        self.assertEqual(recall, 1.0)

        # Undefined recall
        pred = [("car", "inFrontOf", "ego")]
        target = []
        recall = m.calculate_recall(pred, target)
        self.assertEqual(recall, 0.0)

        # Partial recall
        pred = [("car", "inFrontOf", "ego")]
        target = [("car", "inFrontOf", "ego"), ("car", "within_25m", "ego")]
        recall = m.calculate_recall(pred, target)
        self.assertEqual(recall, 0.5)

        # Empty prediction
        pred = []
        target = [("car", "inFrontOf", "ego")]
        recall = m.calculate_recall(pred, target)
        self.assertEqual(recall, 0.0)

        # Empty gt
        pred = [("car", "inFrontOf", "ego")]
        target = []
        recall = m.calculate_recall(pred, target)
        self.assertEqual(recall, 0.0)

        # Both empty
        pred = []
        target = []
        recall = m.calculate_recall(pred, target)
        self.assertEqual(recall, 1.0)

    def test_precision(self):
        m = Metrics()

        # Perfect precision
        pred = [("car", "inFrontOf", "ego")]
        target = [("car", "inFrontOf", "ego")]
        precision = m.calculate_precision(pred, target)
        self.assertEqual(precision, 1.0)

        # Undefined precision
        pred = []
        target = [("car", "inFrontOf", "ego")]
        precision = m.calculate_precision(pred, target)
        self.assertEqual(precision, 0.0)

        # Partial precision
        pred = [("car", "inFrontOf", "ego"), ("car", "inFrontOf", "ego")]
        target = [("car", "inFrontOf", "ego")]
        precision = m.calculate_precision(pred, target)
        self.assertEqual(precision, 0.5)

        # Empty prediction
        pred = []
        target = [("car", "inFrontOf", "ego")]
        precision = m.calculate_precision(pred, target)
        self.assertEqual(precision, 0.0)

        # Empty gt
        pred = [("car", "inFrontOf", "ego")]
        target = []
        precision = m.calculate_precision(pred, target)
        self.assertEqual(precision, 0.0)

        # Both empty
        pred = []
        target = []
        precision = m.calculate_precision(pred, target)
        self.assertEqual(precision, 1.0)

    def test_f1(self):
        m = Metrics()

        # Perfect f1
        pred = [("car", "inFrontOf", "ego")]
        target = [("car", "inFrontOf", "ego")]
        f1 = m.calculate_f1(pred, target)
        self.assertEqual(f1, 1.0)

        # Undefined f1
        pred = []
        target = [("car", "inFrontOf", "ego")]
        f1 = m.calculate_f1(pred, target)
        self.assertEqual(f1, 0.0)

        # Partial f1
        pred = [("car", "inFrontOf", "ego")]
        target = [("car", "inFrontOf", "ego"), ("car", "within_25m", "ego")]
        f1 = m.calculate_f1(pred, target)
        self.assertAlmostEqual(f1, 0.6666, places=3)


    def test_update_heatmaps_triplets(self):
        m = Metrics(entity_names=['vehicle','person','ego'], relationship_names=['in_front_of','within_25m','to_right_of','between_25_and_40m'])

        # Correct initialization
        self.assertIsInstance(m.precision_heatmap, pd.DataFrame)
        self.assertIsInstance(m.recall_heatmap, pd.DataFrame)
        self.assertIsInstance(m.count_matrix_pred, pd.DataFrame)
        self.assertIsInstance(m.count_matrix_gt, pd.DataFrame)

        # Test perfect prediction
        pred = [
            ("vehicle", "in_front_of", "ego"),    
            ("vehicle", "within_25m", "ego")
        ]
        target = [
            ("vehicle", "in_front_of", "ego"),
            ("vehicle", "within_25m", "ego")
        ]
        triplets_metrics = m.update_heatmaps(pred, target)
        triplets_set = set(pred + target)
        self.assertEqual(len(triplets_metrics), len(triplets_set))

        # Test innacurate prediction 1
        pred = [
            ("vehicle", "in_front_of", "ego"),    
            ("vehicle", "within_25m", "ego")
        ]
        target = [
            ("vehicle", "in_front_of", "ego")
        ]
        triplets_metrics = m.update_heatmaps(pred, target)
        triplets_set = set(pred + target)
        # Check triplets metrics size
        self.assertEqual(len(triplets_metrics), len(triplets_set))
        # Check triplets metrics
        self.assertEqual(triplets_metrics['(vehicle,in_front_of,ego)']['tp'], 1)
        self.assertEqual(triplets_metrics['(vehicle,in_front_of,ego)']['fp'], 0)
        self.assertEqual(triplets_metrics['(vehicle,in_front_of,ego)']['fn'], 0)
        self.assertEqual(triplets_metrics['(vehicle,within_25m,ego)']['tp'], 0)
        self.assertEqual(triplets_metrics['(vehicle,within_25m,ego)']['fp'], 1)
        self.assertEqual(triplets_metrics['(vehicle,within_25m,ego)']['fn'], 0)

        # Test innacurate prediction 2
        pred = [
            ("vehicle", "in_front_of", "ego"),
            ("vehicle", "between_25_and_40m", "ego")
        ]
        target = [
            ("vehicle", "in_front_of", "ego"),
            ("vehicle", "within_25m", "ego")
        ]
        triplets_metrics = m.update_heatmaps(pred, target)
        triplets_set = set(pred + target)
        # Check triplets metrics size
        self.assertEqual(len(triplets_metrics), len(triplets_set))
        # Check triplets metrics
        self.assertEqual(triplets_metrics['(vehicle,in_front_of,ego)']['tp'], 1)
        self.assertEqual(triplets_metrics['(vehicle,in_front_of,ego)']['fp'], 0)
        self.assertEqual(triplets_metrics['(vehicle,in_front_of,ego)']['fn'], 0)
        self.assertEqual(triplets_metrics['(vehicle,within_25m,ego)']['tp'], 0)
        self.assertEqual(triplets_metrics['(vehicle,within_25m,ego)']['fp'], 0)
        self.assertEqual(triplets_metrics['(vehicle,within_25m,ego)']['fn'], 1)
        self.assertEqual(triplets_metrics['(vehicle,between_25_and_40m,ego)']['tp'], 0)
        self.assertEqual(triplets_metrics['(vehicle,between_25_and_40m,ego)']['fp'], 1)
        self.assertEqual(triplets_metrics['(vehicle,between_25_and_40m,ego)']['fn'], 0)

    def test_update_heatmaps(self):
        m = Metrics(entity_names=['vehicle','person','ego'], relationship_names=['in_front_of','within_25m','to_right_of','between_25_and_40m'])

        # Correct initialization
        self.assertIsInstance(m.precision_heatmap, pd.DataFrame)
        self.assertIsInstance(m.recall_heatmap, pd.DataFrame)
        self.assertIsInstance(m.count_matrix_pred, pd.DataFrame)
        self.assertIsInstance(m.count_matrix_gt, pd.DataFrame)

        # Test prediction with unknown entity
        pred = [
            ("asdasd", "in_front_of", "ego")  
        ]
        target = [
            ("vehicle", "in_front_of", "ego")
        ]
        _ = m.update_heatmaps(pred, target)
        self.assertEqual(m.precision_heatmap.loc['in_front_of','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['in_front_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_pred.loc['in_front_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_gt.loc['in_front_of','vehicle'], 1)

        m = Metrics(entity_names=['vehicle','person','ego'], relationship_names=['in_front_of','within_25m','to_right_of','between_25_and_40m'])
        pred = [
            ("vehicle", "in_front_of", "asdasd")  
        ]
        target = [
            ("vehicle", "in_front_of", "ego")
        ]
        _ = m.update_heatmaps(pred, target)
        self.assertEqual(m.precision_heatmap.loc['in_front_of','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['in_front_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_pred.loc['in_front_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_gt.loc['in_front_of','vehicle'], 1)

        m = Metrics(entity_names=['vehicle','person','ego'], relationship_names=['in_front_of','within_25m','to_right_of','between_25_and_40m'])
        pred = [
            ("vehicle", "asdasd", "ego")  
        ]
        target = [
            ("vehicle", "in_front_of", "ego")
        ]
        _ = m.update_heatmaps(pred, target)
        self.assertEqual(m.precision_heatmap.loc['in_front_of','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['in_front_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_pred.loc['in_front_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_gt.loc['in_front_of','vehicle'], 1)

        # Test perfect prediction
        m = Metrics(entity_names=['vehicle','person','ego'], relationship_names=['in_front_of','within_25m','to_right_of','between_25_and_40m'])
        pred = [
            ("vehicle", "in_front_of", "ego"),    
            ("vehicle", "within_25m", "ego")
        ]
        target = [
            ("vehicle", "in_front_of", "ego"),
            ("vehicle", "within_25m", "ego")
        ]
        _ = m.update_heatmaps(pred, target)
        self.assertEqual(m.precision_heatmap.loc['in_front_of','vehicle'], 1.0)
        self.assertEqual(m.precision_heatmap.loc['within_25m','vehicle'], 1.0)
        self.assertEqual(m.precision_heatmap.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.precision_heatmap.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['in_front_of','vehicle'], 1.0)
        self.assertEqual(m.recall_heatmap.loc['within_25m','vehicle'], 1.0)
        self.assertEqual(m.recall_heatmap.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(m.count_matrix_pred.loc['in_front_of','vehicle'], 1)
        self.assertEqual(m.count_matrix_pred.loc['within_25m','vehicle'], 1)
        self.assertEqual(m.count_matrix_pred.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_pred.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(m.count_matrix_gt.loc['in_front_of','vehicle'], 1)
        self.assertEqual(m.count_matrix_gt.loc['within_25m','vehicle'], 1)
        self.assertEqual(m.count_matrix_gt.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_gt.loc['between_25_and_40m','vehicle'], 0)

        # Test innacurate prediction 1
        m = Metrics(entity_names=['vehicle','person','ego'], relationship_names=['in_front_of','within_25m','to_right_of','between_25_and_40m'])
        pred = [
            ("vehicle", "in_front_of", "ego"),    
            ("vehicle", "within_25m", "ego")
        ]
        target = [
            ("vehicle", "in_front_of", "ego")
        ]
        _ = m.update_heatmaps(pred, target)
        self.assertEqual(m.precision_heatmap.loc['in_front_of','vehicle'], 1.0)
        self.assertEqual(m.precision_heatmap.loc['within_25m','vehicle'], 0)
        self.assertEqual(m.precision_heatmap.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.precision_heatmap.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['in_front_of','vehicle'], 1.0)
        self.assertEqual(m.recall_heatmap.loc['within_25m','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(m.count_matrix_pred.loc['in_front_of','vehicle'], 1)
        self.assertEqual(m.count_matrix_pred.loc['within_25m','vehicle'], 1)
        self.assertEqual(m.count_matrix_pred.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_pred.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(m.count_matrix_gt.loc['in_front_of','vehicle'], 1)
        self.assertEqual(m.count_matrix_gt.loc['within_25m','vehicle'], 0)
        self.assertEqual(m.count_matrix_gt.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_gt.loc['between_25_and_40m','vehicle'], 0)

        # Test innacurate prediction 2
        m = Metrics(entity_names=['vehicle','person','ego'], relationship_names=['in_front_of','within_25m','to_right_of','between_25_and_40m'])
        pred = [
            ("vehicle", "in_front_of", "ego"),
            ("vehicle", "between_25_and_40m", "ego")
        ]
        target = [
            ("vehicle", "in_front_of", "ego"),
            ("vehicle", "within_25m", "ego")
        ]
        _ = m.update_heatmaps(pred, target)
        self.assertEqual(m.precision_heatmap.loc['in_front_of','vehicle'], 1.0)
        self.assertEqual(m.precision_heatmap.loc['within_25m','vehicle'], 0)
        self.assertEqual(m.precision_heatmap.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.precision_heatmap.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['in_front_of','vehicle'], 1.0)
        self.assertEqual(m.recall_heatmap.loc['within_25m','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(m.count_matrix_pred.loc['in_front_of','vehicle'], 1)
        self.assertEqual(m.count_matrix_pred.loc['within_25m','vehicle'], 0)
        self.assertEqual(m.count_matrix_pred.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_pred.loc['between_25_and_40m','vehicle'], 1)
        self.assertEqual(m.count_matrix_gt.loc['in_front_of','vehicle'], 1)
        self.assertEqual(m.count_matrix_gt.loc['within_25m','vehicle'], 1)
        self.assertEqual(m.count_matrix_gt.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_gt.loc['between_25_and_40m','vehicle'], 0)

    def test_update_heatmaps_compound(self):
        m = Metrics(entity_names=['vehicle','person','ego'], relationship_names=['in_front_of','within_25m','to_right_of','between_25_and_40m'])

        # Correct initialization
        self.assertIsInstance(m.precision_heatmap, pd.DataFrame)
        self.assertIsInstance(m.recall_heatmap, pd.DataFrame)
        self.assertIsInstance(m.count_matrix_pred, pd.DataFrame)
        self.assertIsInstance(m.count_matrix_gt, pd.DataFrame)

        # Test perfect prediction
        pred = [
            ("vehicle", "in_front_of", "ego"),    
            ("vehicle", "within_25m", "ego")
        ]
        target = [
            ("vehicle", "in_front_of", "ego"),
            ("vehicle", "within_25m", "ego")
        ]
        _ = m.update_heatmaps(pred, target)
        self.assertEqual(m.precision_heatmap.loc['in_front_of','vehicle'], 1.0)
        self.assertEqual(m.precision_heatmap.loc['within_25m','vehicle'], 1.0)
        self.assertEqual(m.precision_heatmap.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.precision_heatmap.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['in_front_of','vehicle'], 1.0)
        self.assertEqual(m.recall_heatmap.loc['within_25m','vehicle'], 1.0)
        self.assertEqual(m.recall_heatmap.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(m.count_matrix_pred.loc['in_front_of','vehicle'], 1)
        self.assertEqual(m.count_matrix_pred.loc['within_25m','vehicle'], 1)
        self.assertEqual(m.count_matrix_pred.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_pred.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(m.count_matrix_gt.loc['in_front_of','vehicle'], 1)
        self.assertEqual(m.count_matrix_gt.loc['within_25m','vehicle'], 1)
        self.assertEqual(m.count_matrix_gt.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_gt.loc['between_25_and_40m','vehicle'], 0)

        # Add an innacurate prediction
        pred = [
            ("vehicle", "in_front_of", "ego"),    
            ("vehicle", "within_25m", "ego")
        ]
        target = [
            ("vehicle", "in_front_of", "ego")
        ]
        _ = m.update_heatmaps(pred, target)
        self.assertEqual(m.precision_heatmap.loc['in_front_of','vehicle'], 2.0)
        self.assertEqual(m.precision_heatmap.loc['within_25m','vehicle'], 1.0)
        self.assertEqual(m.precision_heatmap.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.precision_heatmap.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['in_front_of','vehicle'], 2.0)
        self.assertEqual(m.recall_heatmap.loc['within_25m','vehicle'], 1.0)
        self.assertEqual(m.recall_heatmap.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.recall_heatmap.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(m.count_matrix_pred.loc['in_front_of','vehicle'], 2)
        self.assertEqual(m.count_matrix_pred.loc['within_25m','vehicle'], 2)
        self.assertEqual(m.count_matrix_pred.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_pred.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(m.count_matrix_gt.loc['in_front_of','vehicle'], 2)
        self.assertEqual(m.count_matrix_gt.loc['within_25m','vehicle'], 1)
        self.assertEqual(m.count_matrix_gt.loc['to_right_of','vehicle'], 0)
        self.assertEqual(m.count_matrix_gt.loc['between_25_and_40m','vehicle'], 0)

    def test_plot_heatmaps(self):
        m = Metrics(entity_names=['vehicle','person','ego'], relationship_names=['in_front_of','within_25m','to_right_of','between_25_and_40m'])

        # Correct initialization
        self.assertIsInstance(m.precision_heatmap, pd.DataFrame)
        self.assertIsInstance(m.recall_heatmap, pd.DataFrame)
        self.assertIsInstance(m.count_matrix_pred, pd.DataFrame)
        self.assertIsInstance(m.count_matrix_gt, pd.DataFrame)

        # Test perfect prediction
        pred = [
            ("vehicle", "in_front_of", "ego"),    
            ("vehicle", "within_25m", "ego")
        ]
        target = [
            ("vehicle", "in_front_of", "ego"),
            ("vehicle", "within_25m", "ego")
        ]
        _ = m.update_heatmaps(pred, target)
        
        # Add an innacurate prediction
        pred = [
            ("vehicle", "in_front_of", "ego"),    
            ("vehicle", "within_25m", "ego")
        ]
        target = [
            ("vehicle", "in_front_of", "ego")
        ]
        _ = m.update_heatmaps(pred, target)

        avg_p_heatmap, avg_r_heatmap = m.plot_heatmaps()
        self.assertEqual(avg_p_heatmap.loc['in_front_of','vehicle'], 1.0)
        self.assertEqual(avg_p_heatmap.loc['within_25m','vehicle'], 0.5)
        self.assertEqual(avg_p_heatmap.loc['to_right_of','vehicle'], -1)
        self.assertEqual(avg_p_heatmap.loc['between_25_and_40m','vehicle'], -1)
        self.assertEqual(avg_r_heatmap.loc['in_front_of','vehicle'], 1.0)
        self.assertEqual(avg_r_heatmap.loc['within_25m','vehicle'], 1.0)
        self.assertEqual(avg_r_heatmap.loc['to_right_of','vehicle'], -1)
        self.assertEqual(avg_r_heatmap.loc['between_25_and_40m','vehicle'], -1)

        self.assertEqual(avg_p_heatmap.loc['in_front_of','person'], -1)
        self.assertEqual(avg_p_heatmap.loc['within_25m','person'], -1)
        self.assertEqual(avg_p_heatmap.loc['to_right_of','person'], -1)
        self.assertEqual(avg_p_heatmap.loc['between_25_and_40m','person'], -1)
        self.assertEqual(avg_r_heatmap.loc['in_front_of','person'], -1)
        self.assertEqual(avg_r_heatmap.loc['within_25m','person'], -1)
        self.assertEqual(avg_r_heatmap.loc['to_right_of','person'], -1)
        self.assertEqual(avg_r_heatmap.loc['between_25_and_40m','person'], -1)

        # Add an innacurate prediction 2
        pred = [
            ("person", "to_right_of", "ego"),    
            ("person", "between_25_and_40m", "ego"),    
            ("vehicle", "to_right_of", "ego"),
            ("vehicle", "between_25_and_40m", "ego")
        ]
        target = [   
            ("vehicle", "to_right_of", "ego"),
            ("vehicle", "within_25m", "ego")
        ]
        _ = m.update_heatmaps(pred, target)

        avg_p_heatmap, avg_r_heatmap = m.plot_heatmaps()
        self.assertEqual(avg_p_heatmap.loc['in_front_of','vehicle'], 1.0)
        self.assertEqual(avg_p_heatmap.loc['within_25m','vehicle'], 0.5)
        self.assertEqual(avg_p_heatmap.loc['to_right_of','vehicle'], 1)
        self.assertEqual(avg_p_heatmap.loc['between_25_and_40m','vehicle'], 0)
        self.assertEqual(avg_r_heatmap.loc['in_front_of','vehicle'], 1.0)
        self.assertEqual(avg_r_heatmap.loc['within_25m','vehicle'], 0.5)
        self.assertEqual(avg_r_heatmap.loc['to_right_of','vehicle'], 1)
        self.assertEqual(avg_r_heatmap.loc['between_25_and_40m','vehicle'], -1)

        self.assertEqual(avg_p_heatmap.loc['in_front_of','person'], -1)
        self.assertEqual(avg_p_heatmap.loc['within_25m','person'], -1)
        self.assertEqual(avg_p_heatmap.loc['to_right_of','person'], 0)
        self.assertEqual(avg_p_heatmap.loc['between_25_and_40m','person'], 0)
        self.assertEqual(avg_r_heatmap.loc['in_front_of','person'], -1)
        self.assertEqual(avg_r_heatmap.loc['within_25m','person'], -1)
        self.assertEqual(avg_r_heatmap.loc['to_right_of','person'], -1)
        self.assertEqual(avg_r_heatmap.loc['between_25_and_40m','person'], -1)    

    def test_get_tp_fp_fn(self):
        m = Metrics()

        ### Empty
        gt = []
        pred = []
        tp, fp, fn = m.get_tp_fp_fn(pred, gt)
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

        ### Lowercase
        gt = [
            ("a","b","c"),
            ("d","e","f")
        ]
        pred = [
            ("A","B","c"),
            ("D","E","F")
        ]
        tp, fp, fn = m.get_tp_fp_fn(pred, gt)
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

        ### Missing tuples
        gt = [
            ("A","B","C"),
            ("D","E","F")
        ]
        pred = [
            ("A","B","C"),
            ("D","E","F")
        ]
        tp, fp, fn = m.get_tp_fp_fn(pred, gt)
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

        gt = [
            ("A","B","C"),
            ("D","E","F")
        ]
        pred = [
            ("A","B","C")
        ]
        tp, fp, fn = m.get_tp_fp_fn(pred, gt)
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 1)

        gt = [
            ("A","B","C"),
        ]
        pred = [
            ("A","B","C"),
            ("D","E","F")
        ]
        tp, fp, fn = m.get_tp_fp_fn(pred, gt)
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)

        ### Missing tuples & different predictions
        gt = [
            ("A","B","C"),
            ("D","E","F")
        ]
        pred = [
            ("G","H","I"),
            ("D","E","F")
        ]
        tp, fp, fn = m.get_tp_fp_fn(pred, gt)
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)

        gt = [
            ("A","B","C"),
            ("D","E","F")
        ]
        pred = [
            ("G","H","I")
        ]
        tp, fp, fn = m.get_tp_fp_fn(pred, gt)
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 2)

        gt = [
            ("A","B","C"),
        ]
        pred = [
            ("G","H","I"),
            ("D","E","F")
        ]
        tp, fp, fn = m.get_tp_fp_fn(pred, gt)
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 2)
        self.assertEqual(fn, 1)

    def test_calculate_metrics(self):
        m = Metrics()

        ### Empty case
        gt = []
        pred = []
        recall, precision, f1, tp, fp, fn, metrics_dict = m.calculate_metrics(pred, gt)
        self.assertEqual(recall, 1)
        self.assertEqual(precision, 1)
        self.assertEqual(f1, 1)
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

        ### Empty prediction case
        gt = [
            ("A","B","C")
        ]
        pred = []
        recall, precision, f1, tp, fp, fn, metrics_dict = m.calculate_metrics(pred, gt)
        self.assertEqual(recall, 0)
        self.assertEqual(precision, 0)
        self.assertEqual(f1, 0)
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 1)

        ### Empty ground truth case
        gt = []
        pred = [
            ("A","B","C")
        ]
        recall, precision, f1, tp, fp, fn, metrics_dict = m.calculate_metrics(pred, gt)
        self.assertEqual(recall, 0)
        self.assertEqual(precision, 0)
        self.assertEqual(f1, 0)
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)

        ### Perfect case
        gt = [
            ("A","B","C"),
            ("D","E","F")
        ]
        pred = [
            ("A","B","C"),
            ("D","E","F")
        ]
        recall, precision, f1, tp, fp, fn, metrics_dict = m.calculate_metrics(pred, gt)
        self.assertEqual(recall, 1)
        self.assertEqual(precision, 1)
        self.assertEqual(f1, 1)
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

        ### Missing tuples
        gt = [
            ("A","B","C"),
            ("D","E","F")
        ]
        pred = [
            ("A","B","C")
        ]
        recall, precision, f1, tp, fp, fn, metrics_dict = m.calculate_metrics(pred, gt)
        self.assertEqual(recall, 0.5)
        self.assertEqual(precision, 1)
        self.assertEqual(f1, 2/3)
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 1)

        ### Missing tuples & different predictions
        gt = [
            ("A","B","C"),
            ("D","E","F")
        ]
        pred = [
            ("G","H","I"),
            ("D","E","F")
        ]
        recall, precision, f1, tp, fp, fn, metrics_dict = m.calculate_metrics(pred, gt)
        self.assertEqual(recall, 0.5)
        self.assertEqual(precision, 0.5)
        self.assertEqual(f1, 0.5)
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)
