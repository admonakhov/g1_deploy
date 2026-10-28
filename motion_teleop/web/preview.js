/* Local 3D reference preview. This code never changes the robot's stream source. */
(() => {
  const el=id=>document.getElementById(id), canvas=el('preview-canvas'), ctx=canvas.getContext('2d');
  let clip=null, time=0, playing=false, last=0, yaw=-.65, pitch=.2, zoom=1, dragging=null, revision=0;
  const panel=el('preview-panel'), play=el('preview-play'), seek=el('preview-seek');
  const mix=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);
  function frame(){
    let lo=0,hi=clip.times.length-1;
    while(hi-lo>1){const mid=(lo+hi)>>1;if(clip.times[mid]<=time)lo=mid;else hi=mid;}
    const alpha=Math.max(0,Math.min(1,(time-clip.times[lo])/(clip.times[hi]-clip.times[lo])));
    return clip.positions[lo].map((v,i)=>mix(v,clip.positions[hi][i],alpha));
  }
  function draw(){
    const rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
    const w=rect.width,h=rect.height;
    if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    if(!clip){ctx.fillStyle='#90a69b';ctx.font='11px system-ui';ctx.textAlign='center';ctx.fillText('Выберите движение из библиотеки',w/2,h/2);return;}
    const points=frame(), center=[points[0][0],points[0][1],.8], scale=Math.min(w/3.0,h/1.9)*zoom;
    const sy=Math.sin(yaw),cy=Math.cos(yaw),sp=Math.sin(pitch),cp=Math.cos(pitch);
    function project(p){const x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];
      const horizontal=cy*x-sy*y,depth=sy*x+cy*y,vertical=cp*z-sp*depth;
      const perspective=4/Math.max(1,4+cp*depth+sp*z);
      return [w/2+horizontal*scale*perspective,h*.51-vertical*scale*perspective];}
    function line(a,b,color,width){const p=project(a),q=project(b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();}
    const gx=Math.floor(center[0]),gy=Math.floor(center[1]);
    for(let i=-5;i<=5;i++){line([gx+i*.5,gy-2.5,0],[gx+i*.5,gy+2.5,0],'#dce8e4',1);line([gx-2.5,gy+i*.5,0],[gx+2.5,gy+i*.5,0],'#dce8e4',1);}
    line([gx,gy,0],[gx+.5,gy,0],'#bc7168',2);line([gx,gy,0],[gx,gy+.5,0],'#6c9cb8',2);
    const stride=Math.max(1,Math.floor(clip.positions.length/150));
    for(let i=stride;i<clip.positions.length;i+=stride){const a=clip.positions[i-stride][0],b=clip.positions[i][0];line([a[0],a[1],.01],[b[0],b[1],.01],'#b5cfbf',1.5);}
    for(const [a,b] of clip.edges){const name=clip.body_names[b],color=name.startsWith('left_')?'#12b5a4':name.startsWith('right_')?'#518ec7':'#294a46';line(points[a],points[b],color,4);}
    for(const point of points){const p=project(point);ctx.fillStyle='#294a46';ctx.beginPath();ctx.arc(...p,3,0,Math.PI*2);ctx.fill();}
  }
  function update(){play.textContent=playing?'Пауза':'Играть';seek.value=time;el('preview-clock').textContent=time.toFixed(2)+' / '+(clip?.duration_s||0).toFixed(2)+' с';}
  window.showMotionPreview=async function(id,title,options={}){
    const ticket=++revision;el('preview-title').textContent='Загрузка…';panel.hidden=false;if(options.scroll!==false)panel.scrollIntoView({behavior:'smooth',block:'nearest'});
    const response=await fetch('/api/jobs/'+encodeURIComponent(id)+'/preview');
    if(!response.ok)throw Error('Не удалось загрузить предпросмотр');
    const data=await response.json();if(ticket!==revision)return;
    clip=data;time=0;playing=options.autoplay!==false;last=performance.now();el('preview-title').textContent=title;
    seek.max=clip.duration_s;seek.disabled=false;play.disabled=false;el('preview-reset').disabled=false;update();draw();
    window.dispatchEvent(new CustomEvent('motion-preview',{detail:{id}}));
  };
  play.onclick=()=>{if(!clip)return;if(time>=clip.duration_s)time=0;playing=!playing;update();};
  el('preview-reset').onclick=()=>{time=0;playing=false;update();draw();};
  seek.oninput=()=>{time=Number(seek.value);playing=false;update();draw();};
  canvas.onpointerdown=e=>{dragging=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(!dragging)return;yaw+=(e.clientX-dragging[0])*.008;pitch=Math.max(-.6,Math.min(1.1,pitch+(e.clientY-dragging[1])*.005));dragging=[e.clientX,e.clientY];draw();};
  canvas.onpointerup=canvas.onpointercancel=()=>{dragging=null;};
  canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.3,Math.min(3,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
  new ResizeObserver(draw).observe(canvas);
  function tick(now){if(playing&&clip){time=Math.min(clip.duration_s,time+Math.min(.1,(now-last)/1000)*Number(el('preview-speed').value));if(time>=clip.duration_s)playing=false;update();draw();}last=now;requestAnimationFrame(tick);}
  requestAnimationFrame(tick);
})();
