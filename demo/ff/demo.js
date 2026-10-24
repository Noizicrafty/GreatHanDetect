/* Local presentation demo. Presentation fixtures and genuine checkpoint predictions stay distinct. */
'use strict';
let state, selectedId=new URLSearchParams(location.search).get('id'), category='all', order={column:0,asc:false};
const page=location.pathname.split('/').pop(), $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const statusText={new:'На проверке',confirmed:'Подтверждено',rejected:'Отклонено',auto:'Автоматически зарегистрировано'};
const statusColor={new:'amber',confirmed:'green',rejected:'red',auto:'gray'};
const model=e=>e.source==='trained_checkpoint';
const sourceLabel=e=>model(e)?'YOLOv8n · 26 эпох':'DEMO · презентация';
const badge=e=>`<span class="badge ${statusColor[e.status]}">${statusText[e.status]}</span>`;
const actions=e=>e.status==='new'?`<button class="btn danger" data-review="rejected" data-id="${esc(e.id)}">Отклонить</button><button class="btn primary" data-review="confirmed" data-id="${esc(e.id)}">Подтвердить</button>`:badge(e);
async function api(path,data){const r=await fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const value=await r.json();if(!r.ok)throw Error(value.error||'Ошибка запроса');return value;}
function toast(text,error=false){let t=$('#demoToast');if(!t){t=document.createElement('div');t.id='demoToast';t.setAttribute('role','status');document.body.append(t);}t.textContent=text;t.classList.toggle('error',error);t.classList.add('show');setTimeout(()=>t.classList.remove('show'),4500);}
async function refresh(){state=await api('/api/demo/state');render();}
function render(){
  if(state.public_origin)$('.demo-banner b').textContent='Демонстрационный стенд';
  $$('.bell-badge').forEach(b=>{b.textContent=state.pending;b.style.display=state.pending?'grid':'none';});
  const pending=state.events.filter(e=>e.status==='new');
  $('#bellPanel').innerHTML=`<h3 class="panel-title">Уведомления · ${pending.length}</h3>`+(pending.length?pending.map(e=>`<a class="notice-mini" href="04_notifications.html?id=${esc(e.id)}"><img src="${esc(e.image)}" alt="${esc(e.title)}"><div><h4>${esc(e.title)}</h4><p>${esc(e.camera)} · ${esc(e.time)}</p><p>${sourceLabel(e)}</p></div></a>`).join(''):'<p class="incident-sub">Все уведомления обработаны</p>');
  $('#modelState').textContent=state.model_ready?'Модель 26 эпох доступна':'Режим презентации';
  $('#predictOpen').disabled=!state.model_ready;
  $('#webcamOpen').disabled=!state.model_ready;
  if(page==='01_monitoring.html')renderMonitoring();
  if(page==='04_notifications.html')renderNotices();
  if(page==='02_analytics.html')renderAnalytics();
}
function eventCard(e){
  const media=e.clip?`<button class="run-media" data-run="${esc(e.clip)}" data-still="${esc(e.image)}" aria-label="Проиграть демонстрационный GIF" aria-pressed="false"><img alt="${esc(e.title)}" src="${esc(e.image)}"><span class="run-control">▶ GIF</span></button>`:`<a href="04_notifications.html?id=${esc(e.id)}"><img alt="${esc(e.title)}" src="${esc(e.image)}"></a>`;
  return `<article class="incident-card" id="${esc(e.id)}">${media}<div class="incident-body"><div class="incident-head"><div><h3 class="incident-title">${esc(e.title)}</h3><div class="incident-sub">${esc(e.camera)} · ${esc(e.zone)}</div></div><span class="badge ${model(e)?'green':'gray'}">${sourceLabel(e)}</span></div><div class="incident-meta"><div class="meta-box"><small>Время</small><b>${esc(e.time)}</b></div><div class="meta-box"><small>${model(e)?'Confidence детекции':'Уверенность · пример'}</small><b>${Math.round(e.confidence*100)}%</b></div><div class="meta-box"><small>Статус</small><b>${statusText[e.status]}</b></div><div class="meta-box"><small>ID события</small><b>${esc(e.id)}</b></div></div>${model(e)?'<div class="incident-sub" style="margin-top:10px">Объектная детекция кадра. Временной эпизод не подтверждён.</div>':''}<div class="incident-actions">${actions(e)}</div></div></article>`;
}
function renderMonitoring(){
  const es=state.events.filter(e=>e.active);
  $('.incident-grid').innerHTML=es.map(eventCard).join('');
  const heads=$$('.content-grid .card-head .badge');if(heads[0])heads[0].textContent=`Событий: ${es.length}`;if(heads[1])heads[1].textContent=`${state.pending} на проверке`;
  $('.queue').innerHTML=es.map(e=>`<a class="alert-item ${e.severity==='critical'?'critical':''}" href="#${esc(e.id)}"><div class="alert-head">${badge(e)}<span>${esc(e.time)}</span></div><h3>${esc(e.title)}</h3><p>${esc(e.camera)} · ${esc(e.zone)}</p><p>${sourceLabel(e)}</p></a>`).join('');
}
function renderNotices(){
  const pending=state.events.filter(e=>e.status==='new');
  if(!selectedId)selectedId=pending.find(model)?.id||pending.find(e=>e.category==='smoking')?.id||pending[0]?.id;
  const e=state.events.find(e=>e.id===selectedId);
  $('.notice-list').innerHTML=pending.length?pending.map(item=>`<button class="notice-list-item ${item.id===selectedId?'selected':''}" data-select="${esc(item.id)}"><img alt="${esc(item.title)}" src="${esc(item.image)}"><div><strong>${esc(item.title)}</strong><div class="incident-sub">${esc(item.camera)} · ${esc(item.time)}</div><div class="incident-sub">${esc(item.zone)}</div><div class="incident-sub">${sourceLabel(item)}</div></div></button>`).join(''):'<div class="empty-demo"><b>Все уведомления обработаны</b><p>Решения сохранены. Новые кадры можно проверить кнопкой сверху.</p><a href="02_analytics.html" class="btn">Открыть журнал</a></div>';
  $('.notice-page .card-head .badge').textContent=pending.length;
  const head=$$('.notice-page .card-head')[1];if(head)head.innerHTML='<span class="card-title">Форма уведомления</span>'+(e?badge(e):'');
  $('.notice-form-page').innerHTML=e?`<img alt="${esc(e.title)}" src="${esc(e.image)}"><div class="incident-head" style="margin-top:14px"><div><h3 class="incident-title">${esc(e.title)}</h3><div class="incident-sub">${sourceLabel(e)}${model(e)?' · обнаружение объектов на одном кадре':''}</div></div><span class="badge amber">${Math.round(e.confidence*100)}%</span></div><div class="incident-meta"><div class="meta-box"><small>Камера</small><b>${esc(e.camera)}</b></div><div class="meta-box"><small>Зона</small><b>${esc(e.zone)}</b></div><div class="meta-box"><small>Время / дата</small><b>${esc(e.time)} · ${esc(e.date)}</b></div><div class="meta-box"><small>Статус</small><b>${statusText[e.status]}</b></div></div><div class="incident-actions"><a class="btn" href="${esc(e.image)}" download>Скачать кадр</a>${actions(e)}</div>`:'<div class="empty-demo">Выберите событие в списке.</div>';
}
function renderAnalytics(){
  const es=state.events,rejected=es.filter(e=>e.status==='rejected').length,confidence=es.length?Math.round(es.reduce((n,e)=>n+e.confidence,0)/es.length*100):0;
  const vals=[es.length,rejected,confidence+'%',new Set(es.map(e=>e.camera)).size];
  const subs=['События локального журнала','Решение оператора, не оценка модели','Среднее confidence; не точность','Демо-камеры и загруженные кадры'];
  $$('.kpi').forEach((n,i)=>{$('.value',n).textContent=vals[i];$('.sub',n).textContent=subs[i];});
  $$('.kpi .label')[1].textContent='Отклонено оператором';
  const counts={};es.forEach(e=>{const key=e.category;counts[key]=(counts[key]||0)+1;});
  const labels={helmet:'Отсутствие каски',harness:'Страховочный трос',smoking:'Курение / признаки',running:'Бег'};
  $('.bars').innerHTML=Object.entries(counts).map(([k,v])=>`<div class="bar-row"><div>${labels[k]||esc(k)}</div><div class="bar-track"><div class="bar-fill" style="width:${v/Math.max(es.length,1)*100}%"></div></div><div>${v}</div></div>`).join('');
  $('.analytics-grid').style.gridTemplateColumns='1fr';
  if(!$('.filter[data-category="harness"]'))$('.filters').insertAdjacentHTML('beforeend','<button class="filter" data-category="harness">Страховочный трос</button>');
  const rows=[...es].sort((a,b)=>{const key=['time','camera','zone','title','confidence','status'][order.column]||'time';const result=typeof a[key]==='number'?a[key]-b[key]:String(a[key]).localeCompare(String(b[key]),'ru');return order.asc?result:-result;});
  $('#eventsTable tbody').innerHTML=rows.map(e=>`<tr data-category="${esc(e.category)}"><td>${esc(e.time)}</td><td>${esc(e.camera)}</td><td>${esc(e.zone)}</td><td>${esc(e.title)}<div class="incident-sub">${sourceLabel(e)}</div></td><td>${Math.round(e.confidence*100)}%</td><td>${badge(e)}</td><td><a class="clip-link" href="${esc(e.clip||e.image)}" download>Скачать ${e.clip?'GIF':'кадр'}</a></td></tr>`).join('');
  $$('.filter').forEach(b=>b.onclick=()=>{category=b.dataset.category;applyFilter();});$('#searchInput').oninput=applyFilter;
  $('.toolbar .btn').onclick=()=>location.assign('/api/demo/export.xlsx?category='+encodeURIComponent(category)+'&q='+encodeURIComponent($('#searchInput').value));
  $$('#eventsTable th button').forEach((b,i)=>{b.setAttribute('aria-label','Сортировать: '+b.parentElement.textContent.replace('▾','').trim());b.onclick=()=>{order={column:i,asc:order.column===i?!order.asc:true};renderAnalytics();};});
  applyFilter();
}
function applyFilter(){const q=($('#searchInput')?.value||'').toLowerCase();$$('#eventsTable tbody tr').forEach(r=>r.hidden=!((category==='all'||r.dataset.category===category)&&r.textContent.toLowerCase().includes(q)));$$('.filter').forEach(b=>b.classList.toggle('active',b.dataset.category===category));if(!$('#emptyRegistry'))$('#eventsTable').insertAdjacentHTML('afterend','<div id="emptyRegistry" class="empty-demo" hidden>По выбранным условиям события не найдены.</div>');$('#emptyRegistry').hidden=$$('#eventsTable tbody tr').some(r=>!r.hidden);}
function setup(){
  const banner=document.createElement('div');banner.className='demo-banner';banner.innerHTML='<div><b>Локальное демо</b><span>Камеры и нарушения из презентации — примеры. Проверка загруженного кадра — реальная модель, без обучения.</span></div><div class="demo-controls"><span id="modelState">Подключение…</span><button id="predictOpen" class="btn primary" disabled>Проверить кадр</button><a class="btn" href="/api/demo/dataset.zip" title="Только подтверждённые кадры checkpoint; боксы являются псевдометками">Кадры для датасета</a><button id="resetDemo" class="btn">Сбросить примеры</button></div>';
  $('.main-inner').prepend(banner);
  const mobile=document.createElement('nav');mobile.className='mobile-nav';mobile.setAttribute('aria-label','Мобильная навигация');
  mobile.innerHTML=[['03_cameras.html','▦','Камеры'],['01_monitoring.html','▣','Мониторинг'],['02_analytics.html','▤','Аналитика'],['04_notifications.html','♧','Уведомления'],['05_report.html','▧','Отчёт']].map(([url,icon,name])=>`<a href="${url}" class="${url===page?'active':''}"><span aria-hidden="true">${icon}</span>${name}</a>`).join('');document.body.append(mobile);
  $('#bellBtn').onclick=e=>{e.stopPropagation();$('#bellPanel').classList.toggle('open');$('#bellBtn').setAttribute('aria-expanded',$('#bellPanel').classList.contains('open'));};
  document.addEventListener('click',async e=>{
    if(!e.target.closest('.bell-wrap'))$('#bellPanel').classList.remove('open');
    const select=e.target.closest('[data-select]');if(select){selectedId=select.dataset.select;renderNotices();}
    const review=e.target.closest('[data-review]');if(review){review.disabled=true;try{await api('/api/demo/review',{id:review.dataset.id,status:review.dataset.review});await refresh();toast('Решение сохранено в локальном журнале');}catch(err){review.disabled=false;toast(err.message,true);}}
    const run=e.target.closest('[data-run]');if(run){const playing=run.getAttribute('aria-pressed')!=='true';run.setAttribute('aria-pressed',playing);$('img',run).src=playing?run.dataset.run:run.dataset.still;$('.run-control',run).textContent=playing?'Ⅱ Пауза':'▶ GIF';}
  });
  $('#resetDemo').onclick=async()=>{if(!confirm('Восстановить демонстрационные события из презентации? Результаты модели сохранятся.'))return;try{await api('/api/demo/reset',{});selectedId=null;await refresh();toast('Демонстрационные события восстановлены');}catch(e){toast(e.message,true);}};
  $('#predictOpen').onclick=openPredict;
  const webcamButton=document.createElement('button');webcamButton.id='webcamOpen';webcamButton.className='btn primary';webcamButton.textContent='Веб-камера';webcamButton.disabled=true;$('#predictOpen').after(webcamButton);
  webcamButton.onclick=async()=>{try{const module=await import('/webcam.mjs');module.openWebcam();}catch(e){toast('Не удалось открыть веб-камеру: '+e.message,true);}};
  if(page==='05_report.html'){
    const a=$('.card-head a');a.href='/api/demo/report.pdf?download=1';a.textContent='Скачать PDF';a.download='LinQ_Safety_report.pdf';
    $('iframe').src='/api/demo/report.pdf';
  }
  if(page==='03_cameras.html'){
    $$('.cam').forEach((c,i)=>{const img=$('img',c),src=img.src;const btn=document.createElement('button');btn.className='btn';btn.textContent='▶ Демозапись';btn.style.marginTop='10px';let play=false;btn.onclick=()=>{play=!play;img.src=play?`assets/animations/camera_0${i+6}.gif`:src;btn.textContent=play?'Ⅱ Остановить':'▶ Демозапись';};$('.cam-body',c).append(btn);});
    $('.legend').insertAdjacentHTML('beforeend','<span>Демонстрационные записи из архива · подключение к настоящим камерам не выполняется</span>');
  }
}
function openPredict(){
  let dialog=$('#predictDialog');if(!dialog){dialog=document.createElement('dialog');dialog.id='predictDialog';document.body.append(dialog);}
  dialog.innerHTML='<div class="dialog-head"><h2>Проверить кадр моделью</h2><button class="btn" id="closePredict" aria-label="Закрыть">✕</button></div><p class="incident-sub">YOLOv8n · checkpoint после 26 эпох · conf ≥ 0,25. Только локальный инференс. Обучение и LinQ не запускаются.</p><form id="predictForm"><label class="upload-label">Выберите фотографию (JPG, PNG, WEBP, до 10 МБ)<input id="predictFile" type="file" accept="image/jpeg,image/png,image/webp" required></label><div class="incident-actions" style="justify-content:flex-start"><button class="btn primary" id="predictSubmit">Запустить проверку</button><button type="button" class="btn" id="predictExample">Использовать готовый пример</button></div></form><div id="predictionResult" aria-live="polite"></div>';
  $('#closePredict').onclick=()=>dialog.close();dialog.showModal();
  if(state.public_origin)$('#predictDialog .incident-sub').textContent='YOLOv8n · checkpoint после 26 эпох · conf ≥ 0,25. Инференс на сервере. Обучение и LinQ не запускаются.';
  $('#predictExample').onclick=async()=>{try{const r=await fetch('assets/photos/trained_sample.jpg');if(!r.ok)throw Error('Пример не найден');const file=new File([await r.blob()],'trained_sample.jpg',{type:'image/jpeg'});const dt=new DataTransfer();dt.items.add(file);$('#predictFile').files=dt.files;$('#predictForm').requestSubmit();}catch(e){toast(e.message,true);}};
  $('#predictForm').onsubmit=async e=>{e.preventDefault();const f=$('#predictFile').files[0];if(!f)return;if(f.size>10*1024*1024){toast('Размер файла больше 10 МБ',true);return;}const btn=$('#predictSubmit');btn.disabled=true;btn.textContent='Модель обрабатывает кадр…';$('#predictionResult').textContent='Загрузка checkpoint и инференс могут занять несколько секунд.';
    try{const data=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);r.onerror=reject;r.readAsDataURL(f);});const result=await api('/api/demo/predict',{image:data});await refresh();$('#predictionResult').innerHTML=`<img class="prediction-image" src="${esc(result.image)}" alt="Реальные предсказания checkpoint"><p><b>${esc(result.message)}</b></p><p class="incident-sub">${result.prediction.epochs} эпох · ${esc(result.prediction.backend)} · ${result.prediction.detections.length} детекций. Кадровая детекция не подтверждает временной эпизод.</p>${result.event?`<a class="btn primary" href="04_notifications.html?id=${esc(result.event.id)}">Открыть событие</a>`:''}`;}catch(err){$('#predictionResult').textContent=err.message;toast(err.message,true);}finally{btn.disabled=false;btn.textContent='Запустить проверку';}
  };
}
setup();refresh().catch(e=>toast('Сервер недоступен: '+e.message,true));
