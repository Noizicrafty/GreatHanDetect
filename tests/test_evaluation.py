import unittest
import numpy as np
from greathan.evaluation import evaluate
from greathan.tiny_yolo import decode_tiny_yolo


def e(start,end,**kw):return {"camera_id":"c","zone_id":"z","event_type":"danger_zone","start_s":start,"end_s":end,**kw}


class EvaluationTests(unittest.TestCase):
    def test_duplicate_is_false_positive(self):
        report=evaluate([e(1,5),e(1.2,4.8)],[e(1,5)],30)
        self.assertEqual((report["tp"],report["fp"],report["fn"],report["duplicates"]),(1,1,0,1))

    def test_wrong_camera_does_not_match(self):
        report=evaluate([e(1,5,camera_id="other")],[e(1,5)],30)
        self.assertEqual((report["tp"],report["fp"],report["fn"]),(0,1,1))

    def test_matching_is_one_to_one_even_for_overlapping_truth(self):
        report=evaluate([e(1,8),e(1,3)],[e(1,3),e(6,9)],30)
        self.assertEqual(report["tp"],2)
        self.assertEqual(len({m["prediction_index"] for m in report["matches"]}),2)

    def test_empty_negative_clip_has_no_undefined_success_claim(self):
        report=evaluate([],[],20)
        self.assertIsNone(report["precision"])
        self.assertIsNone(report["recall"])

    def test_tiny_decoder_voc_person_and_shape_validation(self):
        raw=np.full((1,125,13,13),-20,dtype=np.float32)
        raw[0,0:4,5,5]=0
        raw[0,4,5,5]=10
        raw[0,5+14,5,5]=10
        result=decode_tiny_yolo(raw,(1,0,0,416,416))
        self.assertEqual(len(result),1)
        self.assertEqual(result[0][0],"person")
        self.assertGreater(result[0][1],.99)
        with self.assertRaises(ValueError):decode_tiny_yolo(np.zeros((1,84,8400)),(1,0,0,416,416))
