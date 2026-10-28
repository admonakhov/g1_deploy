import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from motion_teleop import codex_planner as codex
from motion_teleop.planner import PlannerError,PlannerSettings,plan_schema
from motion_teleop.tests.test_planner import plan
from motion_teleop.trace_worker import stage


class CodexPlannerTests(unittest.TestCase):
    def auth(self):
        return dict(available=True,compatible=True,authenticated=True,binary='/test/codex',version='codex-cli test')

    def test_subscription_probe_does_not_expose_auth_details(self):
        results=[subprocess.CompletedProcess([],0,'Logged in using ChatGPT\n',''),
                 subprocess.CompletedProcess([],0,'codex-cli test\n',''),
                 subprocess.CompletedProcess([],0,'--output-schema --ephemeral --ignore-user-config --sandbox','')]
        with patch.object(codex,'find_codex',return_value='/test/codex'),patch.object(codex.subprocess,'run',side_effect=results):
            result=codex.probe_codex()
            self.assertTrue(result['authenticated']);self.assertTrue(result['compatible'])
        results[0]=subprocess.CompletedProcess([],0,'Logged in using an API key: secret-test-key','')
        with patch.object(codex,'find_codex',return_value='/test/codex'),patch.object(codex.subprocess,'run',side_effect=results):
            result=codex.probe_codex()
            self.assertFalse(result['authenticated']);self.assertNotIn('secret-test-key',json.dumps(result))

    def test_cli_environment_excludes_api_credentials_without_touching_cached_auth(self):
        with patch.dict(os.environ,{k:'fake-secret' for k in codex.API_ENVIRONMENT}):
            env=codex.codex_environment()
            self.assertTrue(all(k not in env for k in codex.API_ENVIRONMENT))
            self.assertEqual(env.get('HOME'),os.environ.get('HOME'))
            self.assertEqual(env.get('CODEX_HOME'),os.environ.get('CODEX_HOME'))

    def test_settings_use_subscription_and_do_not_reuse_api_key(self):
        with patch.object(codex,'find_codex',return_value='/test/codex'):
            settings=PlannerSettings('codex',api_key='private-api-key')
            public,key=settings.snapshot()
            self.assertEqual(public['planner_provider'],'codex');self.assertEqual(key,'')
            self.assertEqual(public['codex_bin'],'/test/codex')
            self.assertEqual(public['codex_model'],'')
            self.assertNotIn('private-api-key',json.dumps(settings.status()))
            settings.update(dict(codex_model='test-codex'))
            self.assertEqual(settings.snapshot()[0]['planner_model'],'test-codex')

    def test_cli_schema_stdin_readonly_no_tools_and_ephemeral_output(self):
        captured=[]
        def run(command,**kwargs):
            captured.append((command,kwargs))
            output=Path(command[command.index('--output-last-message')+1])
            output.write_text(json.dumps(plan()))
            event=dict(type='turn.completed',usage=dict(input_tokens=123,output_tokens=45))
            return subprocess.CompletedProcess(command,0,json.dumps(event)+'\n','')
        with patch.object(codex,'probe_codex',return_value=self.auth()),patch.object(codex.subprocess,'run',side_effect=run):
            value,status=codex._run_codex('Wave the hand',plan_schema(),'test-codex',60)
        self.assertEqual(value,plan());self.assertEqual(status['usage']['input_tokens'],123)
        command,kwargs=captured[0]
        self.assertNotIn('Wave the hand',command);self.assertEqual(kwargs['input'],'Wave the hand')
        self.assertIn('--ignore-user-config',command);self.assertIn('--ephemeral',command)
        self.assertEqual(command[command.index('--sandbox')+1],'read-only')
        self.assertIn('approval_policy="never"',command)
        for feature in ('shell_tool','apps','plugins','hooks','multi_agent'):
            i=command.index(feature);self.assertEqual(command[i-1],'--disable')
        self.assertFalse(Path(command[command.index('--cd')+1]).exists(),'Temporary context must be removed')

    def test_cli_auth_failure_timeout_and_tool_events_fail_explicitly(self):
        bad=dict(available=True,compatible=True,authenticated=False,message='Run codex login')
        with patch.object(codex,'probe_codex',return_value=bad),patch.object(codex.subprocess,'run') as run:
            with self.assertRaisesRegex(PlannerError,'codex login'):codex._run_codex('Wave',{},'',1)
            run.assert_not_called()
        with patch.object(codex,'probe_codex',return_value=self.auth()):
            with patch.object(codex.subprocess,'run',side_effect=subprocess.TimeoutExpired('codex',1)):
                with self.assertRaisesRegex(PlannerError,'timed out'):codex._run_codex('Wave',{},'',1)
            with patch.object(codex.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps(dict(type='item.completed',item=dict(type='command_execution'))),'')):
                with self.assertRaisesRegex(PlannerError,'tool operation'):codex._run_codex('Wave',{},'',1)

    def test_real_trace_stage_compiles_codex_reply_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'request.json').write_text(json.dumps(dict(planner_provider='codex',text='Wave',device='cpu')))
            with patch.object(codex,'_run_codex',return_value=(plan(),dict(version='test',usage={}))),patch.dict('sys.modules',{'torch':None}):
                stage('planner',folder)
            status=json.loads((folder/'planner_status.json').read_text())
            self.assertEqual(status['auth_mode'],'chatgpt');self.assertFalse(status['failed'])
            self.assertTrue((folder/'plan.npz').is_file())
            for file in folder.iterdir():self.assertNotIn(b'api_key',file.read_bytes())

if __name__=='__main__':unittest.main()
