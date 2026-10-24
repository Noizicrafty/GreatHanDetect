from types import SimpleNamespace as NS
import json
import os
import unittest
from unittest.mock import patch
import numpy as np
from greathan.backends import AmqpLinQBackend, BackendError


class Codec:
    TARGET_URI_ATTR_NAME="uri"
    MESSAGE_ID_ATTR_NAME="tpu_pycloud_message_id"
    RESPONSE_ROUTING_KEYS_ATTR_NAME="response_routing_keys"
    PyTask=staticmethod(lambda x:x)
    serialize=staticmethod(lambda x:b"test-request")
    output={"scores":np.ones((1,2),np.float32)}
    worker={"status_code":0,"worker_pid":"qa","inference_time":.01}
    deserialize=classmethod(lambda cls,body:(cls.output,(),cls.worker))


class Channel:
    def __init__(self,mode):self.mode=mode;self.responses=[];self.published=[]
    def queue_declare(self,**kw):self.declaration=kw;return NS(method=NS(queue="exclusive-test"))
    def queue_bind(self,**kw):pass
    def basic_publish(self,**kw):
        self.published.append(kw)
        request_id=kw["properties"].headers[Codec.MESSAGE_ID_ATTR_NAME]
        if self.mode=="wrong_then_right":self.responses.append((NS(),NS(headers={Codec.MESSAGE_ID_ATTR_NAME:"other"}),b"response"))
        if self.mode!="timeout":self.responses.append((NS(),NS(headers={Codec.MESSAGE_ID_ATTR_NAME:request_id}),b"response"))
    def basic_get(self,*args,**kwargs):return self.responses.pop(0) if self.responses else (None,None,None)
    def consume(self,*args,**kwargs):
        while True:
            yield self.basic_get()


class Connection:
    def __init__(self,channel):self.test_channel=channel;self.is_open=True
    def channel(self):return self.test_channel
    def process_data_events(self,**kw):pass
    def close(self):self.is_open=False


class AmqpTests(unittest.TestCase):
    def setUp(self):
        self.environment=patch.dict(os.environ,{"LINQ_AMQP_USER":"test","LINQ_AMQP_PASSWORD":"secret"})
        self.environment.start();self.addCleanup(self.environment.stop)
        Codec.output={"scores":np.ones((1,2),np.float32)}
        Codec.worker={"status_code":0,"worker_pid":"qa"}
        self.model={"remote_program":"/auto/cache/team/program.tpu","inputs":{"image":{"shape":[1,3,2,2]}},"outputs":{"scores":{"shape":[1,2]}}}
        self.input={"image":np.zeros((1,3,2,2),np.float32)}

    def backend(self,mode="normal"):
        self.channel=Channel(mode);self.connection=Connection(self.channel)
        pika=NS(BlockingConnection=lambda p:self.connection,ConnectionParameters=lambda **kw:kw,
                PlainCredentials=lambda a,b:None,BasicProperties=lambda **kw:NS(**kw))
        return AmqpLinQBackend({"host":"example","timeout_s":.01},pika_module=pika,codec=Codec)

    def test_request_contract_and_unique_ids(self):
        backend=self.backend()
        outputs,timing=backend.infer(self.model,self.input)
        backend.infer(self.model,self.input)
        self.assertEqual(outputs["scores"].shape,(1,2))
        props=[p["properties"].headers for p in self.channel.published]
        self.assertNotEqual(props[0][Codec.MESSAGE_ID_ATTR_NAME],props[1][Codec.MESSAGE_ID_ATTR_NAME])
        self.assertEqual(props[0][Codec.TARGET_URI_ATTR_NAME],"file:///auto/cache/team/program.tpu?raw=False")
        self.assertEqual(json.loads(props[0][Codec.RESPONSE_ROUTING_KEYS_ATTR_NAME]),["exclusive-test"])
        self.assertTrue(self.channel.declaration["exclusive"])
        backend.close();self.assertFalse(self.connection.is_open)

    def test_foreign_response_is_ignored(self):
        backend=self.backend("wrong_then_right")
        _,timing=backend.infer(self.model,self.input)
        self.assertEqual(timing["ignored_responses"],1)
        backend.close()

    def test_timeout_closes_connection(self):
        backend=self.backend("timeout")
        with self.assertRaises(BackendError):backend.infer(self.model,self.input)
        self.assertFalse(self.connection.is_open)

    def test_worker_error_is_not_reported_as_empty_scene(self):
        backend=self.backend();Codec.worker={"status_code":7,"error_message":"secret"}
        with self.assertRaisesRegex(BackendError,"status 7") as error:backend.infer(self.model,self.input)
        self.assertNotIn("secret",str(error.exception))

    def test_bad_output_names_fail(self):
        backend=self.backend();Codec.output={"wrong":np.ones((1,2),np.float32)}
        with self.assertRaises(ValueError):backend.infer(self.model,self.input)
        self.assertFalse(self.connection.is_open)

    def test_wrong_input_shape_fails_before_publish(self):
        backend=self.backend()
        with self.assertRaises(ValueError):backend.infer(self.model,{"image":np.zeros((1,2),np.float32)})
        self.assertEqual(self.channel.published,[])

    def test_nan_output_fails(self):
        backend=self.backend();Codec.output={"scores":np.full((1,2),np.nan,np.float32)}
        with self.assertRaises(ValueError):backend.infer(self.model,self.input)

    def test_remote_path_rejects_windows_and_traversal(self):
        for path in ["C:/test.tpu","/etc/model.tpu","/auto/cache/../etc/model.tpu","/auto/cache/team/model.onnx"]:
            with self.subTest(path=path),self.assertRaises(ValueError):AmqpLinQBackend.program_uri(path)
