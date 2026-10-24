from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from greathan.video import VideoSource


class VideoTests(unittest.TestCase):
    def test_realtime_discards_late_frames_but_offline_preserves_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'test.mp4'
            writer=cv2.VideoWriter(str(file),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,64))
            for i in range(30):writer.write(np.full((64,64,3),i,np.uint8))
            writer.release()
            signal=threading.Event()
            with patch('greathan.video.time.monotonic',return_value=100):
                source=VideoSource({'source':str(file),'mode':'realtime'},signal)
            with patch('greathan.video.time.monotonic',return_value=102):
                item=source.read()
            self.assertGreaterEqual(item[1],1.75)
            self.assertGreater(source.dropped,15)
            source.close()
            offline=VideoSource({'source':str(file),'mode':'offline'},threading.Event())
            self.assertEqual(offline.read()[1],0)
            self.assertAlmostEqual(offline.read()[1],.1,places=3)
            offline.close()

    def test_local_file_is_not_silently_a_live_camera(self):
        with self.assertRaisesRegex(ValueError,'network stream'):
            VideoSource({'source':'file.mp4','mode':'live'},threading.Event())
