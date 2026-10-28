const {test}=require('node:test');
const assert=require('node:assert/strict');
const {encodeWav}=require('../web/voice.js');
test('WAV is mono 16-bit 16 kHz and resamples 16/44.1/48 kHz',()=>{
 for(const rate of [16000,44100,48000]){
  const data=encodeWav([new Float32Array(rate).fill(.5)],rate),view=new DataView(data);
  assert.equal(data.byteLength,32044);assert.equal(view.getUint32(24,true),16000);assert.equal(view.getUint16(22,true),1);assert.equal(view.getUint16(34,true),16);
  assert.equal(view.getInt16(44,true),16384);assert.equal(view.getUint32(40,true),32000);
 }
});
test('multiple chunks preserve order; PCM saturation is correct',()=>{
 const view=new DataView(encodeWav([new Float32Array([2]),new Float32Array([-2,0])],16000));
 assert.deepEqual([44,46,48].map(i=>view.getInt16(i,true)),[32767,-32768,0]);
});
test('capture cannot exceed the server duration limit',()=>{
 assert.equal(encodeWav([new Float32Array(21*16000)],16000).byteLength,20*32000+44);
});
test('audio worklet batches samples and flushes the final partial chunk',()=>{
 const vm=require('node:vm'),fs=require('node:fs');let Processor;const messages=[];
 class Base{constructor(){this.port={postMessage:message=>messages.push(message)}}}
 vm.runInNewContext(fs.readFileSync(require.resolve('../web/voice-worklet.js'),'utf8'),{AudioWorkletProcessor:Base,registerProcessor:(name,ctor)=>{assert.equal(name,'motion-voice-capture');Processor=ctor}});
 const capture=new Processor();for(let i=0;i<17;i++)capture.process([[new Float32Array(128).fill(.25)]]);
 assert.equal(messages[0].samples.length,2048);capture.port.onmessage({data:'flush'});
 assert.equal(messages[1].samples.length,128);assert.equal(messages[2].flushed,true);
 assert.equal(messages[1].samples[0],.25);
});
