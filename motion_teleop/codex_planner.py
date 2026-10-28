"""Subscription-backed motion planning through the official authenticated Codex CLI."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from .planner import PlannerError, _trace_compile, compile_api_plan, plan_schema


# These are separate from the CLI's cached ChatGPT login, managed by Codex itself.
API_ENVIRONMENT = (
    'OPENAI_API_KEY', 'CODEX_API_KEY', 'CODEX_ACCESS_TOKEN', 'OPENAI_ACCESS_TOKEN',
    'OPENAI_BASE_URL', 'OPENAI_MODEL', 'OPENAI_ORG_ID', 'OPENAI_PROJECT_ID',
    'OPENAI_IDENTITY_TOKEN_FILE', 'OPENAI_SERVICE_ACCOUNT_ID',
)
DISABLED_FEATURES = (
    'shell_tool', 'unified_exec', 'apps', 'plugins', 'hooks', 'multi_agent',
    'browser_use', 'browser_use_external', 'computer_use', 'in_app_browser',
    'image_generation', 'view_image', 'code_mode', 'code_mode_host', 'memories',
)


def codex_environment():
    env=dict(os.environ)
    for name in API_ENVIRONMENT:
        env.pop(name,None)
    # The invocation is an independent ephemeral planner, not a continuation of
    # whichever desktop/terminal task happened to start the motion server.
    env.pop('CODEX_THREAD_ID',None)
    env.pop('CODEX_INTERNAL_ORIGINATOR_OVERRIDE',None)
    return env


def find_codex(binary=None):
    if binary:
        candidate=shutil.which(str(binary))
        if candidate:
            return candidate
        path=Path(binary).expanduser()
        if path.is_file() and os.access(path,os.X_OK):
            return str(path.absolute())
        return None
    found=shutil.which('codex')
    if found:
        return found
    bundled=Path('/usr/lib/chatgpt/resources/codex')
    return str(bundled) if bundled.is_file() and os.access(bundled,os.X_OK) else None


def probe_codex(binary=None):
    executable=find_codex(binary)
    result=dict(available=bool(executable),authenticated=False,auth_mode=None,
                compatible=False,checked=True,binary=executable,version=None)
    if not executable:
        return dict(result,message='Codex CLI is not installed or its path is invalid')
    try:
        env=codex_environment()
        status=subprocess.run([executable,'login','status'],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=10,env=env)
        # Never return raw login output: some versions print account/key details.
        chatgpt=status.returncode==0 and 'using chatgpt' in (status.stdout+status.stderr).lower()
        result.update(authenticated=chatgpt,auth_mode='chatgpt' if chatgpt else None)
        version=subprocess.run([executable,'--version'],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=10,env=env)
        first=version.stdout.strip().splitlines()
        result['version']=first[0][:120] if first and first[0].startswith('codex') else None
        help_result=subprocess.run([executable,'exec','--help'],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=10,env=env)
        result['compatible']=help_result.returncode==0 and all(flag in help_result.stdout for flag in ('--output-schema','--ephemeral','--ignore-user-config','--sandbox'))
        if not result['compatible']:
            result['message']='Update Codex CLI: this integration needs --output-schema, --ephemeral and --ignore-user-config'
        elif not chatgpt:
            result['message']='Run codex login and choose Sign in with ChatGPT, then check sign-in again'
        else:
            result['message']='Signed in with ChatGPT; subscription access will be checked on generation'
    except (OSError,subprocess.SubprocessError):
        result['message']='Could not check Codex CLI; check its installation and sign-in'
    return result


def codex_command(executable,directory,model=''):
    directory=Path(directory)
    command=[executable,'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check',
             '--sandbox','read-only','--color','never','--json','--cd',str(directory),
             '--output-schema',str(directory/'schema.json'),
             '--output-last-message',str(directory/'reply.json'),
             '-c','approval_policy="never"','-c','project_doc_max_bytes=0',
             '-c','web_search="disabled"','-c','model_reasoning_effort="low"']
    for name in DISABLED_FEATURES:
        command.extend(['--disable',name])
    if model:
        command.extend(['--model',model])
    command.append('-')  # Prompt goes through stdin, never command-line arguments.
    return command


def _run_codex(prompt,schema,model,timeout,binary=None):
    auth=probe_codex(binary)
    if not auth.get('available') or not auth.get('compatible') or not auth.get('authenticated'):
        raise PlannerError(auth['message'])
    with tempfile.TemporaryDirectory(prefix='g1-codex-plan-') as directory:
        folder=Path(directory)
        (folder/'schema.json').write_text(json.dumps(schema))
        try:
            completed=subprocess.run(codex_command(auth['binary'],folder,model),input=prompt,
                capture_output=True,text=True,timeout=timeout,env=codex_environment())
        except subprocess.TimeoutExpired:
            raise PlannerError('Codex planning timed out; current reference playback continues') from None
        except OSError:
            raise PlannerError('Could not start Codex CLI') from None
        if completed.returncode:
            # Do not persist CLI stderr or credentials. A failed subscription
            # request must never silently fall back to API-key billing.
            raise PlannerError('Codex could not complete the plan. Check subscription limits, model access and codex login status')
        usage={}
        for line in completed.stdout.splitlines():
            try:
                event=json.loads(line)
            except ValueError:
                continue
            if not isinstance(event,dict):
                continue
            if event.get('type')=='turn.failed':
                raise PlannerError('Codex planning failed; check subscription limits and model access')
            item=event.get('item') or {}
            if not isinstance(item,dict):
                item={}
            if item.get('type') in ('command_execution','mcp_tool_call','web_search','file_change'):
                raise PlannerError('Codex attempted a tool operation; only a JSON motion plan is accepted')
            if event.get('type')=='turn.completed':
                raw=event.get('usage',{})
                usage={k:v for k,v in raw.items() if k in ('input_tokens','cached_input_tokens','output_tokens') and type(v) is int}
        reply=folder/'reply.json'
        if not reply.is_file() or reply.stat().st_size>2*1024*1024:
            raise PlannerError('Codex returned no usable structured plan')
        try:
            return json.loads(reply.read_text()),dict(version=auth['version'],usage=usage)
        except (ValueError,UnicodeError):
            raise PlannerError('Codex returned an invalid JSON motion plan') from None


def predict_codex_plan(config,output):
    import numpy as np
    output=Path(output)
    _,_,system=_trace_compile()
    minimum=2*int(config.get('boundary_frames',8))+4
    model=config.get('codex_model','')
    prompt=(system+'\nReturn the named event objects defined by the output schema. '
            f'Choose {minimum}..512 frames at 20 FPS. End values are exclusive 4-frame patch indices; '
            'the final end must equal ceil(frames/4). Merge adjacent identical states. '
            'Root height means pelvis height above ground, not head or torso height. Neutral G1 standing pelvis height is about 0.8 m: root_height=1. Unless a crouch, jump or travel is requested, keep root_vertical=1 and both feet in stable support. '
            'Do not use tools, inspect files, browse, execute commands or modify anything. '
            'This is a self-contained motion-planning request. Output only the JSON plan.\n'
            'Motion description:\n'+config['text'])
    began=time.monotonic()
    for attempt in range(2):
        value,info=_run_codex(prompt,plan_schema(minimum),model,float(config.get('planner_timeout_s',180.)),config.get('codex_bin'))
        try:
            obj,probability,known,status=compile_api_plan(value,minimum)
            break
        except (ValueError,TypeError,KeyError,IndexError):
            if attempt:
                raise PlannerError('Codex plan failed TRACE timeline validation twice; no motion was generated') from None
            prompt+='\nThe previous plan failed timeline validation. Check frame count, category ranges, strictly increasing patch ends and complete coverage.'
    (output/'planner_response.txt').write_text(json.dumps(obj,ensure_ascii=False)+'\n')
    (output/'plan.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    np.savez(output/'plan.npz',events=probability,known=known,frames=obj['frames'])
    status.update(provider='codex',model=model or 'Codex CLI default',auth_mode='chatgpt',
                  seconds=time.monotonic()-began,attempts=attempt+1,**info)
    (output/'planner_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(f"Codex subscription plan accepted: {obj['frames']} frames, {len(obj['events'])} events",flush=True)
