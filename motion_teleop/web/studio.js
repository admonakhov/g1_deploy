/* Motion Studio: one selected reference source; previews never send robot goals. */
const $=id=>document.getElementById(id);
let snapshot=null,socket,lastJobs='',plannerInitialized=false,autoAuthRequested=false;
let selectedPreviewId=null,previewRequested=false,libraryFilter='all';
const hzHistory=[];
const stages={planner:'Планирование',embedding:'Подготовка текста',diffusion:'Создание движения',smooth:'Сглаживание',contact:'Контакты стоп',training:'Экспорт для обучения'};
const sourceNames={video:'Видео',trace:'TRACE',idle:'Стойка'};
const stateNames={ready:'Готово',running:'Генерация',queued:'В очереди',failed:'Ошибка',cancelled:'Отменено',interrupted:'Прервано'};
const deviceNames={cpu:'CPU',xpu:'Intel GPU',auto:'Авто'};
let generationSubmitting=false;
function icon(name){const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('class','icon');svg.setAttribute('aria-hidden','true');const use=document.createElementNS(svg.namespaceURI,'use');use.setAttribute('href','#i-'+name);svg.append(use);return svg}
async function api(path,data){const res=await fetch('/api/'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const body=await res.json();if(!res.ok)throw Error(body.error||res.statusText);return body}
async function action(fn){$('error').textContent='';try{return await fn()}catch(e){$('error').textContent=e.message}}
function promptCount(){$('prompt-count').textContent=$('prompt').value.length+' / 2000'}
$('prompt').addEventListener('input',promptCount);
for(const source of ['video','trace','idle'])$(source).onclick=()=>action(async()=>render(await api('source',{source})));
async function generateMotion(){if(generationSubmitting||snapshot?.jobs?.some(j=>['running','queued'].includes(j.state)))throw Error('Движение уже генерируется. Дождитесь завершения или скажите «отмени генерацию».');if(!$('prompt').value.trim())throw Error('Введите описание движения');generationSubmitting=true;try{await applyPlanner();await api('generate',{text:$('prompt').value,steps:Number($('steps').value),seed:Number($('seed').value),autoplay:$('autoplay').checked,contact_refine:$('refine').value})}finally{generationSubmitting=false}}
$('generate').onclick=()=>action(generateMotion);
$('cancel').onclick=()=>action(()=>api('cancel',{}));
$('stream-pause').onclick=()=>action(async()=>render(await api('playback',{action:'pause'})));
$('stream-resume').onclick=()=>action(async()=>render(await api('playback',{action:'resume'})));
window.addEventListener('motion-voice',event=>{
 event.detail.finished=(async()=>{try{const command=event.detail.command;switch(command){
  case 'generate':await generateMotion();break;
  case 'cancel':render(await api('cancel',{}));break;
  case 'pause':render(await api('cancel',{}));if(snapshot?.bridge.source!=='trace')throw Error('Генерация отменена. Пауза движения доступна для TRACE; для смены источника скажите «стойка» или «включи видео».');render(await api('playback',{action:'pause'}));break;
  case 'resume':render(await api('playback',{action:'resume'}));break;
  case 'video':case 'idle':render(await api('source',{source:command}));break;
  case 'replay':{const job=snapshot?.jobs?.find(j=>j.state==='ready');if(!job)throw Error('Нет готового движения. Сначала создайте его.');render(await api('source',{source:'trace',motion_id:job.id}));break}
  default:throw Error('Неизвестная голосовая команда');
 }event.detail.complete({generate:'Генерация запущена',cancel:'Генерация отменена',pause:'Траектория на паузе: удерживается текущий референс',resume:'Воспроизведение продолжено',video:'Выбран видеопоток',idle:'Выбрана исходная стойка',replay:'Воспроизвожу последнее готовое движение'}[command])}catch(error){event.detail.fail(error.message)}})();
});
function plannerFields(){const provider=$('planner-provider').value;$('openai-fields').hidden=provider!=='openai';$('planner-clear').hidden=provider!=='openai';$('codex-fields').hidden=provider!=='codex'}
$('planner-provider').onchange=plannerFields;
async function applyPlanner(){const data={provider:$('planner-provider').value,model:$('openai-model').value.trim(),codex_model:$('codex-model').value.trim()};if($('openai-key').value)data.api_key=$('openai-key').value.trim();const result=await api('planner',data);$('openai-key').value='';render(result);$('planner-notice').textContent='Настройки применены';return result}
$('planner-save').onclick=()=>action(applyPlanner);
async function checkCodex(){const b=$('codex-check');b.disabled=true;try{render(await api('codex/status',{}))}finally{b.disabled=false}}
$('codex-check').onclick=()=>action(checkCodex);
$('planner-clear').onclick=()=>action(async()=>{$('openai-key').value='';render(await api('planner',{clear_api_key:true}));$('planner-notice').textContent='Ключ удалён из памяти сервера'});
$('settings-link').onclick=()=>{$('planner-panel').open=true;document.querySelectorAll('.nav-link').forEach(a=>a.classList.toggle('current',a.id==='settings-link'))};
document.querySelectorAll('.nav-link:not(#settings-link)').forEach(a=>a.onclick=()=>document.querySelectorAll('.nav-link').forEach(link=>link.classList.toggle('current',link===a)));
document.querySelectorAll('[data-text]').forEach(b=>b.onclick=()=>{$('prompt').value=b.dataset.text;promptCount();$('prompt').focus()});
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{libraryFilter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(other=>other.classList.toggle('active',other===b));renderLibrary(true)});
$('motion-search').oninput=()=>renderLibrary(true);
window.addEventListener('motion-preview',event=>{selectedPreviewId=event.detail.id;document.querySelectorAll('.job').forEach(card=>card.classList.toggle('is-selected',card.dataset.id===selectedPreviewId))});
function renderPlanner(p){
 if(!p)return;
 if(!plannerInitialized){$('planner-provider').value=p.provider;$('openai-model').value=p.model;$('codex-model').value=p.codex_model||'';plannerFields();plannerInitialized=true}
 $('planner-summary').textContent=p.provider==='codex'?'Codex · подписка ChatGPT':p.provider==='openai'?'OpenAI · '+p.model:'Qwen · локально';
 const c=p.codex||{},status=$('codex-auth-status');status.classList.toggle('ready',!!c.authenticated);
 status.textContent=c.authenticated?'● Подписка подключена':!c.available?'Codex CLI не найден':!c.checked?'Проверяем вход в Codex…':!c.compatible?'Обновите Codex CLI':'Нужен вход через ChatGPT: codex login';
 status.title=c.version||'';
 $('openai-key-status').textContent=p.api_key_configured?'● API-ключ подключён':'Введите API-ключ OpenAI';$('openai-key-status').classList.toggle('ready',!!p.api_key_configured);
 if(p.provider==='codex'&&!c.checked&&!autoAuthRequested){autoAuthRequested=true;checkCodex().catch(()=>{status.textContent='Не удалось проверить вход. Попробуйте ещё раз.'})}
}
function renderChart(hz){
 hzHistory.push(Number(hz)||0);if(hzHistory.length>40)hzHistory.shift();
 const values=hzHistory.length>1?hzHistory:[hzHistory[0],hzHistory[0]],maximum=Math.max(55,...values),points=values.map((v,i)=>[i/(values.length-1)*240,40-Math.max(0,v)/maximum*30]);
 const path=points.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+','+p[1].toFixed(1)).join(' ');
 $('stream-line').setAttribute('d',path);$('stream-area').setAttribute('d',path+' L240,42 L0,42 Z');
}
function render(data){
 snapshot=data;renderPlanner(data.planner);
 window.dispatchEvent(new CustomEvent('motion-status',{detail:data}));
 const b=data.bridge,jobs=data.jobs||[];
 for(const s of ['video','trace','idle'])$(s).classList.toggle('active',b.source===s);
 $('trace').disabled=!jobs.some(j=>j.state==='ready');
 $('hz').textContent=Number(b.output_hz||0).toFixed(1);$('stream-target').textContent='Целевая частота · '+b.target_hz+' Гц';renderChart(b.output_hz);
 $('selected-source').textContent=sourceNames[b.source]||'Ожидание';
 $('source-description').textContent=b.paused?'Пауза · удержание референса':b.playing?'Воспроизведение движения':b.source==='trace'?'Удержание конечной позы':b.source==='video'?(b.reference_status==='fallback'?'Ожидание видеопотока':'Видео → референс'):'Исходная WBC-стойка';
 $('destination').textContent=data.udp_destination;
 $('camera').textContent=b.video_age_s!==null&&b.video_age_s<.75?'Подключена':'Ожидание';
 $('mode').textContent=b.paused?'Пауза':b.playing?'Воспроизведение':b.source==='trace'?'Конечная поза':b.reference_status==='fallback'?'Исходная стойка':'Телеоп';
 $('stream-pause').disabled=b.source!=='trace'||!b.playing;$('stream-resume').disabled=!b.paused;
 const active=jobs.find(j=>j.state==='running'||j.state==='queued');
 $('generation').textContent=active?(stages[active.progress?.stage]||'Запуск'):'Готов к запросу';
 $('generate').disabled=!!active;$('cancel').disabled=!active;
 $('models').textContent=data.models_ready?'TRACE готов · '+(deviceNames[data.device_info?.device||data.device]||data.device)+'. Embedding и генерация работают локально.':'Модели TRACE ещё не установлены.';
 $('progress').style.width=(b.duration_s?Math.min(100,100*b.position_s/b.duration_s):0)+'%';
 $('time').textContent=b.duration_s?b.position_s.toFixed(1)+' / '+b.duration_s.toFixed(1)+' с':'Движение ещё не выбрано';
 if(b.output_error)$('error').textContent=b.output_error;
 renderLibrary();
 if(!previewRequested){const first=jobs.find(j=>j.state==='ready');if(first){previewRequested=true;showMotionPreview(first.id,first.text,{autoplay:false,scroll:false}).catch(error=>{$('preview-title').textContent='Не удалось загрузить движение';$('error').textContent=error.message})}}
}
function jobButton(label,symbol,handler,className=''){const button=document.createElement('button');button.type='button';button.className=className;if(symbol)button.append(icon(symbol));button.append(document.createTextNode(label));button.onclick=()=>action(handler);return button}
function renderLibrary(force=false){
 if(!snapshot)return;
 const jobs=snapshot.jobs||[],query=$('motion-search').value.toLocaleLowerCase().trim();
 const signature=JSON.stringify(jobs);if(!force&&signature===lastJobs)return;lastJobs=signature;
 const filtered=jobs.filter(j=>(!query||(j.text||'').toLocaleLowerCase().includes(query))&&(libraryFilter==='all'||libraryFilter==='ready'&&j.state==='ready'||libraryFilter==='running'&&['running','queued'].includes(j.state)));
 $('library-count').textContent=jobs.filter(j=>j.state==='ready').length;
 $('jobs').replaceChildren();
 if(!filtered.length){const empty=document.createElement('div');empty.className='empty-state';empty.append(icon('cube'));const p=document.createElement('p');p.textContent=jobs.length?'Движения не найдены':'Ваше первое движение начинается с идеи';empty.append(p);$('jobs').append(empty);return}
 filtered.forEach((j,index)=>{
  const card=document.createElement('article');card.className='job'+(selectedPreviewId===j.id?' is-selected':'');card.dataset.id=j.id;
  const main=document.createElement('div');main.className='job-main';
  const mark=document.createElement('span');mark.className='motion-icon'+(index%3===1?' blue':index%3===2?' purple':'');mark.append(icon('motion'));
  const copy=document.createElement('div');copy.className='job-copy';const title=document.createElement('p');title.className='job-title';title.textContent=j.text;title.title=j.text;
  const meta=document.createElement('div');meta.className='job-meta';const planner=j.planner_provider==='codex'?'Codex':j.planner_provider==='openai'?'OpenAI':'Qwen';
  meta.textContent=(j.duration_s?j.duration_s.toFixed(1)+' с · ':'')+planner+(j.created?' · '+new Intl.DateTimeFormat('ru-RU',{day:'numeric',month:'short'}).format(new Date(j.created*1000)):'');
  if(j.result)meta.title='Генерация: '+j.result.seconds.toFixed(0)+' с · '+j.result.device;
  const status=document.createElement('div');status.className='job-state'+(['running','queued'].includes(j.state)?' running':j.state==='ready'?'':' failed');status.append(document.createElement('i'));status.append(document.createTextNode(j.state==='running'?(stages[j.progress?.stage]||'Запуск'):stateNames[j.state]||j.state));
  if(j.error)status.title=j.error;
  copy.append(title,meta,status);main.append(mark,copy);
  const actions=document.createElement('div');actions.className='job-actions';
  if(j.state==='ready'){
   actions.append(jobButton('3D','cube',()=>showMotionPreview(j.id,j.text)));
   actions.append(jobButton('В деплой','arrow',()=>api('source',{source:'trace',motion_id:j.id}).then(render),'send-motion'));
   if(j.result?.training_export){const download=document.createElement('a');download.href='/api/jobs/'+encodeURIComponent(j.id)+'/training.npz';download.title='Скачать NPZ для обучения';download.setAttribute('aria-label','Скачать NPZ для обучения');download.setAttribute('download','');download.append(icon('download'));download.append(document.createTextNode('NPZ'));actions.append(download)}
  }
  const log=jobButton('','file',async()=>{const res=await fetch('/api/jobs/'+encodeURIComponent(j.id)+'/log');if(!res.ok)throw Error('Не удалось открыть лог');$('log').textContent=await res.text();$('log').style.display='block';$('log').scrollIntoView({behavior:'smooth',block:'nearest'})},'icon-action');log.title='Лог генерации';log.setAttribute('aria-label','Лог генерации');actions.append(log);
  card.append(main,actions);$('jobs').append(card);
 });
}
function connect(){
 socket=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws/poses');
 socket.onopen=()=>{$('connection-label').textContent='Поток онлайн';$('connection').classList.remove('off')};
 socket.onmessage=event=>{const message=JSON.parse(event.data);if(message.type==='status')render(message);else if(message.status)render(message.status)};
 socket.onclose=()=>{$('connection-label').textContent='Переподключение';$('connection').classList.add('off');setTimeout(connect,1500)};
 socket.onerror=()=>socket.close();
}
promptCount();connect();
