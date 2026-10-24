"""Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from common import letterbox,unletterbox_xyxy
from decoder import tensor_nxc,decode_objects,decode_pose

class PreprocessorTests(unittest.TestCase):
    def test_shape_pad_rgb(self):
        frame=np.zeros((240,320,3),np.uint8)
        frame[:]=(0,0,255)
        tensor,meta=letterbox(frame,640)
        self.assertEqual(tensor.shape,(1,3,640,640))
        self.assertAlmostEqual(float(tensor[0,0,100,100]),1.0)
        self.assertAlmostEqual(float(tensor[0,1,100,100]),0.0)
        self.assertEqual(tensor.dtype,np.float32)
    def test_round_trip_boxes(self):
        frame=np.zeros((200,400,3),np.uint8)
        _,info=letterbox(frame,640)
        raw=np.array([[80,192,240,256]],np.float32)
        box=unletterbox_xyxy(raw,info)[0]
        self.assertTrue(np.allclose(box,[50,20,150,60]))
    def test_object_decode(self):
        arr=np.zeros((1,9,2),np.float32)
        arr[0,:4,0]=[100,110,50,30]
        arr[0,4,0]=.91
        arr[0,:4,1]=[100,110,50,30]
        arr[0,4,1]=.60
        out=decode_objects(arr,(1,0,0,640,640),['cigarette','vape','cigar','pipe','smoke'],.3,.5)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0][0],'cigarette')
    def test_pose_decode(self):
        arr=np.zeros((1,56,1),np.float32)
        arr[0,:4,0]=[160,160,100,120]
        arr[0,4,0]=.98
        arr[0,5:,0]=.5
        out=decode_pose(arr,(1,0,0,640,640),.3,.5)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0][2].shape,(17,2))
        self.assertEqual(out[0][3].shape,(17,))
    def test_bad_shape_fails(self):
        with self.assertRaises(ValueError):tensor_nxc(np.zeros((1,44,8400)),56)
if __name__=='__main__':unittest.main()
