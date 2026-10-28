const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

function harness({permission,fetcher,execute=true}={}){
 const elements=new Map(),listeners=new Map(),nodes=[],tracks=[],requests=[],commands=[];
 function element(id){if(!elements.has(id))elements.set(id,{textContent:'',value:'',hidden:false,dataset:{},style:{},checked:id==='voice-execute'&&execute,classList:{toggle(){}},setAttribute(){},dispatchEvent(){}});return elements.get(id)}
 const track={stopped:false,stop(){this.stopped=true}};tracks.push(track);
 const stream={getTracks:()=>[track]};
 class Context{constructor(){this.sampleRate=16000;this.audioWorklet={addModule:async()=>{}};this.destination={}}createMediaStreamSource(){return{connect(){},disconnect(){}}}async resume(){}async close(){this.closed=true}}
 class Node{constructor(){nodes.push(this);this.port={postMessage:message=>{if(message==='flush')this.port.onmessage({data:{flushed:true}})},close(){}}}connect(){}disconnect(){}}
 class UIEvent{constructor(type,init={}){this.type=type;this.detail=init.detail}}
 const window={isSecureContext:true,AudioContext:Context,AudioWorkletNode:Node,addEventListener:(type,handler)=>listeners.set(type,handler),dispatchEvent:event=>{if(event.type==='motion-voice'){commands.push(event.detail);event.detail.finished=Promise.resolve();event.detail.complete('Выполнено')}else listeners.get(event.type)?.(event)}};
 const context={window,document:{getElementById:element},navigator:{mediaDevices:{getUserMedia:permission||(async()=>stream)}},AudioContext:Context,AudioWorkletNode:Node,Event:UIEvent,CustomEvent:UIEvent,AbortController,setTimeout,clearTimeout,Float32Array,DataView,ArrayBuffer,fetch:async(url,options)=>{requests.push({url,options});return fetcher?fetcher(url,options):{ok:true,json:async()=>({text:'подними правую руку',command:'generate'})}}};
 vm.runInNewContext(fs.readFileSync(require.resolve('../web/voice.js'),'utf8'),context);
 listeners.get('motion-status')({detail:{voice:{available:true}}});
 return{element,listeners,nodes,tracks,requests,commands,stream};
}
test('recording closes microphone and sends one local WAV; recognized command executes once',async()=>{
 const h=harness();await h.element('voice-record').onclick();
 h.nodes[0].port.onmessage({data:{samples:new Float32Array(8000).fill(.1)}});
 await h.element('voice-record').onclick();
 assert.equal(h.tracks[0].stopped,true);assert.equal(h.requests.length,1);
 assert.equal(h.requests[0].url,'/api/voice/transcribe');assert.equal(h.requests[0].options.headers['Content-Type'],'audio/wav');
 assert.equal(h.element('prompt').value,'подними правую руку');assert.equal(h.commands.length,1);
 assert.equal(h.element('voice-status').textContent,'Выполнено');assert.equal(h.element('voice-label').textContent,'Голосовая команда');
});
test('dictation mode fills the prompt without executing commands',async()=>{
 const h=harness({execute:false});await h.element('voice-record').onclick();h.nodes[0].port.onmessage({data:{samples:new Float32Array(8000)}});await h.element('voice-record').onclick();
 assert.equal(h.element('prompt').value,'подними правую руку');assert.equal(h.commands.length,0);
});
test('a short one-word utterance ends automatically after silence',async()=>{
 const h=harness();await h.element('voice-record').onclick();
 h.nodes[0].port.onmessage({data:{samples:new Float32Array(2048).fill(.1)}});
 for(let i=0;i<11;i++)h.nodes[0].port.onmessage({data:{samples:new Float32Array(2048)}});
 await new Promise(resolve=>setTimeout(resolve,10));
 assert.equal(h.requests.length,1);assert.equal(h.commands.length,1);assert.equal(h.tracks[0].stopped,true);
});
test('cancel while awaiting microphone permission stops a late stream and never uploads audio',async()=>{
 let grant;const permission=new Promise(resolve=>{grant=resolve}),h=harness({permission:()=>permission});
 const starting=h.element('voice-record').onclick();await h.element('voice-cancel').onclick();grant(h.stream);await starting;
 assert.equal(h.tracks[0].stopped,true);assert.equal(h.requests.length,0);assert.equal(h.commands.length,0);assert.equal(h.element('voice-label').textContent,'Голосовая команда');
});
test('permission rejection returns to idle with a useful message',async()=>{
 const h=harness({permission:async()=>{const error=Error('denied');error.name='NotAllowedError';throw error}});await h.element('voice-record').onclick();
 assert.match(h.element('voice-status').textContent,/Доступ к микрофону запрещён/);assert.equal(h.element('voice-record').disabled,false);assert.equal(h.requests.length,0);
});
test('failed recognition never executes a motion command',async()=>{
 const h=harness({fetcher:async()=>({ok:false,json:async()=>({error:'Не удалось распознать речь'})})});await h.element('voice-record').onclick();h.nodes[0].port.onmessage({data:{samples:new Float32Array(8000)}});await h.element('voice-record').onclick();
 assert.equal(h.commands.length,0);assert.match(h.element('voice-status').textContent,/Не удалось распознать/);assert.equal(h.tracks[0].stopped,true);
});
