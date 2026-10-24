"""TPU execution wrapper based on documented pytpu API."""
from contextlib import ExitStack

class TPUDualSession:
    def __init__(self, detector_program, pose_program, det_input, pose_input, det_output, pose_output):
        import pytpu as tpu
        devices=tpu.Device.list_devices()
        if not devices: raise RuntimeError('No TPU device. Check /dev/tpu* and driver permissions.')
        self._stack=ExitStack()
        try:
            self.device=self._stack.enter_context(tpu.Device.open(devices[0]))
            self.det_program=self._stack.enter_context(self.device.load(str(detector_program)))
            self.pose_program=self._stack.enter_context(self.device.load(str(pose_program)))
            self.det_infer=self._stack.enter_context(self.det_program.inference())
            self.pose_infer=self._stack.enter_context(self.pose_program.inference())
        except Exception:
            self._stack.close()
            raise
        self.det_input,self.pose_input=det_input,pose_input
        self.det_output,self.pose_output=det_output,pose_output

    @staticmethod
    def _choose(result,expected):
        if expected in result: return result[expected]
        if len(result)==1:
            print('WARNING: TPU output name differs:',list(result),'->',expected)
            return next(iter(result.values()))
        raise ValueError(f'Expected TPU output {expected!r}, found {list(result)}.')
    def detect(self,arr): return self._choose(self.det_infer.sync({self.det_input:arr}),self.det_output)
    def pose(self,arr): return self._choose(self.pose_infer.sync({self.pose_input:arr}),self.pose_output)
    def close(self): self._stack.close()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
