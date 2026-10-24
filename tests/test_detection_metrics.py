import unittest
from greathan.detection_metrics import detection_report


class DetectionMetricTests(unittest.TestCase):
    def test_duplicate_boxes_count_as_false_positive(self):
        records=[{'image':'a','truth':[[0,[0,0,10,10]]],'predictions':[[0,.9,[0,0,10,10]],[0,.8,[0,0,10,10]]]}]
        report=detection_report(records,['cigarette'])['per_class'][0]
        self.assertEqual((report['tp'],report['fp'],report['fn']),(1,1,0))
        self.assertEqual(report['precision'],.5)
        self.assertEqual(report['recall'],1)

    def test_wrong_class_and_failed_request_are_not_successes(self):
        records=[{'image':'a','truth':[[0,[0,0,10,10]]],'predictions':[[1,.9,[0,0,10,10]]]},
                 {'image':'b','truth':[[0,[0,0,10,10]]],'predictions':[],'error':'TimeoutError'}]
        report=detection_report(records,['cigarette','vape'])
        self.assertEqual(report['failed_images'],1)
        self.assertEqual(report['per_class'][0]['fn'],2)
        self.assertIsNone(report['per_class'][1]['recall'])
        self.assertEqual(report['map50'],0)

    def test_operating_threshold_and_ap_are_distinct(self):
        records=[{'image':'a','truth':[[0,[0,0,10,10]]],'predictions':[[0,.2,[0,0,10,10]]]}]
        report=detection_report(records,['cigarette'],confidence=.25)
        self.assertEqual(report['map50'],1)
        self.assertEqual(report['per_class'][0]['tp'],0)
