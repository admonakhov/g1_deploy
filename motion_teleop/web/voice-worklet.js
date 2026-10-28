/* Capture only. Outputs stay silent: microphone audio is never monitored. */
class MotionVoiceCapture extends AudioWorkletProcessor {
 constructor(){super();this.buffer=new Float32Array(2048);this.used=0;this.port.onmessage=event=>{if(event.data==='flush'){this.flush();this.port.postMessage({flushed:true})}}}
 flush(){if(!this.used)return;const chunk=this.buffer.slice(0,this.used);this.port.postMessage({samples:chunk},[chunk.buffer]);this.used=0}
 process(inputs){const channel=inputs[0]?.[0];if(channel)for(const sample of channel){this.buffer[this.used++]=sample;if(this.used===this.buffer.length)this.flush()}return true}
}
registerProcessor('motion-voice-capture',MotionVoiceCapture);
