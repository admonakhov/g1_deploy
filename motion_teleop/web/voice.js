/* Local microphone -> bounded mono WAV -> local Vosk. No cloud speech API. */
(()=>{
 function encodeWav(chunks,sampleRate){
  const count=chunks.reduce((n,c)=>n+c.length,0),samples=new Float32Array(count);let offset=0;
  for(const chunk of chunks){samples.set(chunk,offset);offset+=chunk.length}
  const rate=16000,frames=Math.min(rate*20,Math.floor(count*rate/sampleRate)),bytes=new ArrayBuffer(44+frames*2),view=new DataView(bytes);
  const ascii=(at,text)=>{for(let i=0;i<text.length;i++)view.setUint8(at+i,text.charCodeAt(i))};
  ascii(0,'RIFF');view.setUint32(4,36+frames*2,true);ascii(8,'WAVE');ascii(12,'fmt ');view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);view.setUint32(24,rate,true);view.setUint32(28,rate*2,true);view.setUint16(32,2,true);view.setUint16(34,16,true);ascii(36,'data');view.setUint32(40,frames*2,true);
  for(let i=0;i<frames;i++){const from=i*sampleRate/rate,to=(i+1)*sampleRate/rate;let sum=0;for(let j=Math.floor(from);j<Math.ceil(to);j++)sum+=(samples[j]||0)*(Math.min(to,j+1)-Math.max(from,j));const value=Math.max(-1,Math.min(1,sum/(to-from)));view.setInt16(44+i*2,Math.round(value*(value<0?32768:32767)),true)}
  return bytes;
 }
 if(typeof module!=='undefined'&&module.exports){module.exports={encodeWav};return}
 const el=id=>document.getElementById(id),button=el('voice-record'),cancel=el('voice-cancel'),status=el('voice-status'),label=el('voice-label');
 let state='idle',session=0,stream,context,node,input,chunks=[],samples=0,speech=0,lastSound=0,timer,request,flushDone;
 let ready=false;
 const supported=!!(window.isSecureContext&&navigator.mediaDevices?.getUserMedia&&window.AudioContext&&window.AudioWorkletNode);
 function notice(text,error=false){status.textContent=text;status.classList.toggle('voice-error',error)}
 function setState(value){state=value;button.classList.toggle('recording',value==='recording');button.setAttribute('aria-pressed',String(value==='recording'));button.disabled=!supported||!ready||['requesting','recognizing','executing'].includes(value);cancel.hidden=['idle','executing'].includes(value);label.textContent=value==='recording'?'Завершить запись':value==='recognizing'?'Распознаю…':value==='executing'?'Выполняю…':'Голосовая команда';el('voice-level').style.width='0%'}
 async function release(){clearTimeout(timer);if(node){node.disconnect();node.port.onmessage=null;node.port.close();node=null}if(input){input.disconnect();input=null}stream?.getTracks().forEach(track=>track.stop());stream=null;if(context){const old=context;context=null;await old.close().catch(()=>{})}}
 async function finish(discard=false){
  if(state==='idle'||state==='executing')return;
  if(!discard&&state!=='recording')return;
  const current=++session,wasRecording=state==='recording';
  request?.abort();request=null;
  if(discard){await release();if(current===session){chunks=[];setState('idle');notice('Запись отменена')}return}
  if(!wasRecording)return;
  setState('recognizing');notice('Распознаю русскую речь локально…');
  const rate=context.sampleRate,execute=el('voice-execute').checked;
  if(node)await new Promise(resolve=>{flushDone=resolve;node.port.postMessage('flush');setTimeout(resolve,200)});
  if(current!==session)return;
  const recording=chunks;chunks=[];await release();
  if(current!==session)return;
  try{
   if(samples/rate<.2)throw Error('Запись слишком короткая. Повторите команду.');
   request=new AbortController();const timeout=setTimeout(()=>request?.abort(),35000);
   let response;try{response=await fetch('/api/voice/transcribe',{method:'POST',headers:{'Content-Type':'audio/wav'},body:encodeWav(recording,rate),signal:request.signal})}finally{clearTimeout(timeout)}
   const result=await response.json();if(!response.ok)throw Error(result.error||'Не удалось распознать команду');if(current!==session)return;
   el('voice-transcript').textContent='«'+result.text+'»';el('voice-transcript').hidden=false;
   if(result.command==='generate'){el('prompt').value=result.text;el('prompt').dispatchEvent(new Event('input',{bubbles:true}))}
   if(!execute){notice(result.command==='generate'?'Текст готов. Можно исправить его и нажать «Сгенерировать».':'Распознана команда. Включите выполнение и повторите запись.');return}
   setState('executing');const event=new CustomEvent('motion-voice',{detail:{...result,complete:message=>notice(message),fail:message=>notice(message,true)}});window.dispatchEvent(event);
   // The shared studio handler owns requests and the normal generation settings.
   await event.detail.finished;
  }catch(error){if(current===session)notice(error.name==='AbortError'?'Распознавание прервано. Попробуйте ещё раз.':error.message,true)}
  finally{if(current===session){request=null;setState('idle')}}
 }
 async function start(){
  if(state!=='idle')return;const current=++session;chunks=[];samples=speech=0;lastSound=0;setState('requesting');notice('Разрешите доступ к микрофону…');
  try{
   const acquired=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});
   if(current!==session){acquired.getTracks().forEach(track=>track.stop());return}stream=acquired;
   context=new AudioContext({sampleRate:16000});await context.audioWorklet.addModule('/voice-worklet.js?v=20261028-voice2');if(current!==session)return;
   node=new AudioWorkletNode(context,'motion-voice-capture');input=context.createMediaStreamSource(stream);const rate=context.sampleRate;
   node.port.onmessage=event=>{if(event.data.flushed){flushDone?.();flushDone=null;return}const chunk=event.data.samples;if(!chunk)return;chunks.push(chunk);samples+=chunk.length;if(state!=='recording')return;const seconds=samples/rate,rms=Math.sqrt(chunk.reduce((sum,v)=>sum+v*v,0)/chunk.length);el('voice-level').style.width=Math.min(100,rms*1200)+'%';if(rms>.012){speech+=chunk.length/rate;lastSound=seconds}notice('Слушаю · '+seconds.toFixed(1)+' с · скажите команду по-русски');if(speech>.08&&seconds-lastSound>1.3)finish();else if(seconds>6&&speech===0)finish(true).then(()=>notice('Не слышно речи. Проверьте микрофон и попробуйте снова.',true))};
   input.connect(node);node.connect(context.destination);await context.resume();if(current!==session)return;setState('recording');notice('Слушаю. Например: «Помаши правой рукой».');timer=setTimeout(()=>finish(),20000);
  }catch(error){if(current!==session)return;await release();if(current===session){setState('idle');notice(error.name==='NotAllowedError'?'Доступ к микрофону запрещён. Разрешите его в настройках браузера.':error.name==='NotFoundError'?'Микрофон не найден. Подключите его и повторите.':'Не удалось включить микрофон: '+error.message,true)}}
 }
 button.onclick=()=>state==='recording'?finish():start();cancel.onclick=()=>finish(true);
 window.addEventListener('pagehide',()=>{session++;request?.abort();release()});
 window.addEventListener('keydown',event=>{if(event.key==='Escape'&&state!=='idle')finish(true)});
 window.addEventListener('motion-status',event=>{ready=!!event.detail.voice?.available;if(state==='idle'){setState('idle');if(!supported)notice('Микрофон требует localhost или HTTPS и браузер с Web Audio.',true);else if(!ready)notice('Установите русский голосовой ввод: python -m motion_teleop.setup_voice',true);else if(!status.dataset.ready){status.dataset.ready='true';notice('Русский · локально · запись до 20 секунд')}}});
 setState('idle');
})();
