"""Unavailable and ended inputs keep initial references alive; no real camera."""
from unittest import TestCase
from unittest.mock import Mock
import numpy as np
from video_teleop.stream_capture import RecoveringCapture
from video_teleop.teleop_stream import ContinuousPosePublisher, initial_reference

class SourceRecoveryTests(TestCase):
    def test_failed_open_retries_and_recovers_without_skipping_first_frame(self):
        closed=Mock();closed.isOpened.return_value=False
        opened=Mock();opened.isOpened.return_value=True;opened.get.return_value=30.
        image=np.zeros((10,20,3),dtype=np.uint8)
        opened.read.side_effect=[(True,image),(False,None)]
        opener=Mock(side_effect=[closed,opened])
        reader=RecoveringCapture('test.mp4',retry_interval=0.,opener=opener)
        try:
            self.assertFalse(reader.ready());closed.release.assert_called_once()
            self.assertTrue(reader.ready());self.assertIs(reader.read()[1],image)
            self.assertFalse(reader.read()[0]);self.assertEqual(reader.failure_reason,'source_ended')
            self.assertFalse(reader.ready());self.assertEqual(opener.call_count,2) # EOF does not replay the file.
        finally:reader.close()
        opened.release.assert_called_once()

    def test_empty_input_and_opencv_exception_remain_retryable(self):
        opened=Mock();opened.isOpened.return_value=True;opened.get.return_value=float('nan');opened.read.return_value=(False,None)
        reader=RecoveringCapture('empty.mp4',retry_interval=0.,opener=Mock(side_effect=[OSError('unavailable'),opened]))
        try:
            self.assertFalse(reader.ready());self.assertFalse(reader.ready())
            self.assertFalse(reader.eof);self.assertEqual(reader.fps,25.)
        finally:reader.close()

    def test_input_failure_immediately_returns_initial_instead_of_holding_pose(self):
        transport=Mock();initial=initial_reference()
        publisher=ContinuousPosePublisher(transport,initial,'test',loss_grace=.5)
        observed=dict(initial,status='tracking',active_joint_mask=[True]*29,capture_monotonic_s=10.,processing_completed_monotonic_s=10.)
        publisher.publish(observed)
        self.assertEqual(publisher.reference.sample(10.,100.)['status'],'tracking')
        publisher.use_initial('source_unavailable')
        out=publisher.reference.sample(10.01,100.01)
        self.assertEqual(out['status'],'fallback');self.assertEqual(out['fallback_reason'],'source_unavailable')
        self.assertEqual(out['joint_positions'],initial['joint_positions'])
