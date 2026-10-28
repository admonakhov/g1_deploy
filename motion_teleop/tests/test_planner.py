import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import numpy as np

from motion_teleop import planner
from motion_teleop.trace_worker import stage

KEY='test-only-openai-key-never-a-real-secret'

def plan():
    return dict(frames=100,events=[dict(zip(planner.EVENT_FIELDS,[25,1,1,1,1,0,0]))])

def response(value=None,**extra):
    return dict(id='resp_test',status='completed',output=[dict(type='message',content=[dict(type='output_text',text=json.dumps(value or plan()))])],usage=dict(input_tokens=200,output_tokens=50,total_tokens=250),**extra)


class PlannerTests(unittest.TestCase):
    def test_credentials_are_ephemeral_and_settings_atomic(self):
        settings=planner.PlannerSettings(api_key=KEY)
        self.assertNotIn(KEY,json.dumps(settings.status()))
        settings.update(dict(provider='openai',model='gpt-4.1-mini'))
        public,key=settings.snapshot()
        self.assertEqual(key,KEY);self.assertNotIn(KEY,json.dumps(public))
        with self.assertRaises(ValueError):settings.update(dict(provider='bad',api_key='other'))
        self.assertEqual(settings.snapshot()[1],KEY)
        settings.update(dict(clear_api_key=True))
        with self.assertRaisesRegex(ValueError,'key is missing'):settings.snapshot()
        settings.update(dict(provider='local'))
        self.assertEqual(settings.snapshot()[1],'')
        self.assertIn('Qwen',settings.snapshot()[0]['planner_model'])

    def test_exact_trace_compilation_and_unknown_states(self):
        value=plan();value['events'][0]['left_hand']=-1
        native,probability,known,status=planner.compile_api_plan(value)
        self.assertFalse(status['failed']);self.assertEqual(native['events'][0],[25,1,1,1,1,-1,0])
        self.assertEqual(probability.shape,(128,16));self.assertFalse(known[:25,8:12].any())
        np.testing.assert_allclose(probability[:25,8:12],.25)
        for broken in (dict(frames=100,events=[dict(plan()['events'][0],end=24)]),
                       dict(frames=100,events=[dict(plan()['events'][0],left_support=2)]),
                       dict(frames=100,events=[dict(plan()['events'][0],end=True)]),
                       dict(frames=10,events=[dict(plan()['events'][0],end=3)])):
            with self.assertRaises((ValueError,TypeError)):planner.compile_api_plan(broken)

    def test_openai_http_contract_and_no_credentials_in_files(self):
        payload=response();request_log=[]
        def open_request(request,timeout):
            request_log.append((request,timeout))
            return io.BytesIO(json.dumps(payload).encode())
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,OPENAI_API_KEY=KEY),patch.object(planner,'urlopen',side_effect=open_request):
            folder=Path(directory)
            planner.predict_openai_plan(dict(text='Помаши правой рукой',planner_model='gpt-4.1-mini',boundary_frames=8),folder)
            self.assertEqual(len(request_log),1)
            request,timeout=request_log[0];body=json.loads(request.data)
            self.assertEqual(request.full_url,'https://api.openai.com/v1/responses')
            self.assertEqual(request.get_header('Authorization'),'Bearer '+KEY)
            self.assertFalse(body['store']);self.assertEqual(body['input'],'Помаши правой рукой')
            self.assertEqual(body['text']['format']['type'],'json_schema');self.assertTrue(body['text']['format']['strict'])
            self.assertEqual(body['text']['format']['schema']['properties']['frames']['minimum'],20)
            self.assertEqual(timeout,90.)
            self.assertEqual(json.loads((folder/'plan.json').read_text())['frames'],100)
            with np.load(folder/'plan.npz',allow_pickle=False) as data:
                self.assertEqual(data['events'].shape,(128,16));self.assertEqual(data['frames'].item(),100)
            self.assertEqual(json.loads((folder/'planner_status.json').read_text())['provider'],'openai')
            for file in folder.iterdir():self.assertNotIn(KEY.encode(),file.read_bytes())

    def test_semantic_errors_retry_once_then_fail(self):
        broken=plan();broken['events'][0]['end']=24
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,OPENAI_API_KEY=KEY):
            with patch.object(planner,'_request',side_effect=[response(broken),response()]) as request:
                planner.predict_openai_plan(dict(text='Wave'),Path(directory))
                self.assertEqual(request.call_count,2)
            with patch.object(planner,'_request',return_value=response(broken)) as request:
                with self.assertRaisesRegex(planner.PlannerError,'twice'):planner.predict_openai_plan(dict(text='Wave'),Path(directory))
                self.assertEqual(request.call_count,2)

    def test_api_failure_is_redacted_and_not_retried(self):
        for error in (HTTPError(planner.API_URL,401,KEY,{},io.BytesIO(KEY.encode())),URLError(KEY)):
            with patch.object(planner,'urlopen',side_effect=error):
                with self.assertRaises(planner.PlannerError) as caught:planner._request({},KEY,1.)
                self.assertNotIn(KEY,str(caught.exception))
        for result in (dict(status='incomplete',output=[]),dict(status='completed',output=[]),
                       dict(status='completed',output=[dict(type='message',content=[dict(type='refusal',refusal='no')])])):
            with self.assertRaises(planner.PlannerError):planner.response_text(result)

    def test_stage_does_not_import_or_load_the_local_language_model(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,OPENAI_API_KEY=KEY):
            folder=Path(directory);(folder/'request.json').write_text(json.dumps(dict(planner_provider='openai',text='Wave',device='cpu')))
            with patch.object(planner,'_request',return_value=response()),patch.dict('sys.modules',{'torch':None}):
                stage('planner',folder)
            self.assertTrue((folder/'plan.npz').is_file())

    def test_remote_preflight_requires_only_embedding_and_motion_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);embedding=root/'text_embedding';embedding.mkdir()
            for name in ('generator_boundary.pt','generation_stats.npz'):(root/name).write_bytes(b'fake')
            for name in ('config.json','tokenizer.json','tokenizer_config.json','model.safetensors'):(embedding/name).write_bytes(b'fake')
            (embedding/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'a':'model.safetensors'})))
            planner.check_motion_models(root,'openai')
            with self.assertRaises(FileNotFoundError):planner.check_motion_models(root,'local')

if __name__=='__main__':unittest.main()
