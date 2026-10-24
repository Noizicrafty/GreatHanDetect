import {mapDetections,mouthMatches,Evidence,OrderedFrames} from './webcam-logic.mjs';
let current=null;
const cameraError=e=>({NotAllowedError:'Доступ к камере запрещён. Разрешите камеру в настройках сайта и нажмите «Запустить».',NotFoundError:'Веб-камера не найдена. Подключите её и повторите запуск.',NotReadableError:'Камера занята другим приложением или недоступна.',OverconstrainedError:'Выбранная камера недоступна. Выберите другую.'}[e.name]||e.message||'Не удалось запустить камеру');
export function openWebcam(){
  if(current){current.dialog.showModal();return;}
  if(!document.querySelector('#webcamStyle')){const s=document.createElement('link');s.id='webcamStyle';s.rel='stylesheet';s.href='/webcam.css';document.head.append(s);}
  const dialog=document.createElement('dialog');dialog.id='webcamDialog';
  dialog.innerHTML=`<div class="dialog-head"><h2>Веб-камера · сигарета у рта</h2><button class="btn" id="webcamClose" aria-label="Закрыть веб-камеру">✕</button></div>
  <div class="webcam-toolbar"><label>Камера <select id="webcamDevice" aria-label="Выбрать камеру"><option value="">Камера по умолчанию</option></select></label><label>Обработка <select id="webcamBackend" aria-label="Вычислитель"><option value="cpu">CPU сервера</option><option value="linq" disabled>LinQ TPU · проверка доступности</option></select></label><label>Предел FPS <select id="webcamRate" aria-label="Предел FPS"><option value="2">2</option><option value="5" disabled>5</option><option value="10" disabled>10</option></select></label><button class="btn primary" id="webcamStart">Запустить</button><button class="btn" id="webcamStop" disabled>Остановить</button></div>
  <div class="webcam-grid"><section><h3><span class="webcam-dot"></span>Прямой эфир</h3><div class="webcam-stage"><video id="webcamVideo" autoplay muted playsinline></video><span id="webcamPlaceholder">Нажмите «Запустить» и разрешите камеру</span></div></section><section><h3>Последний обработанный кадр</h3><div class="webcam-stage"><canvas id="webcamResult" width="960" height="540"></canvas></div></section></div>
  <div id="webcamStatus" class="webcam-status" role="status" aria-live="polite">Камера выключена</div>
  <div class="webcam-bottom"><details class="webcam-details"><summary>О проверке</summary>
  <p id="webcamBackendLabel">YOLOv8n · 26 эпох · CPU сервера + контур губ в браузере</p>
  <p id="webcamMetrics">Первый запуск модели может занять до минуты. Затем веса остаются в памяти.</p>
  <p>Жёлтый — сигарета · голубой — область губ · красный — совпадение. Срабатывание: 3 кадра подряд, минимум 0,5 с.</p>
  <p>Проверяется расположение сигареты возле губ, а не факт курения. Подойдите ближе, держите лицо в кадре. Профиль, перекрытия и маленькая сигарета могут давать пропуски.</p>
  <p>После запуска отдельные кадры отправляются на сервер. Звук не запрашивается, поток не записывается. Сохранение выбранного кадра в журнал включает отдельную CPU-проверку. Обучение не запускается.</p>
  </details><button class="btn" id="webcamSave" disabled>Сохранить срабатывание</button></div>`;
  document.body.append(dialog);dialog.showModal();
  const el=id=>dialog.querySelector('#'+id),video=el('webcamVideo'),canvas=el('webcamResult'),ctx=canvas.getContext('2d');
  const controllers=new Set();
  let stream=null,worker=null,timer=null,generation=0,running=false,evidence=new Evidence(),lastImage=null,errors=0,saveBusy=false,targetRate=2,completed=[];
  let facePending=null,faceReady=false,faceUnavailable=false;
  const status=(text,kind='')=>{el('webcamStatus').textContent=text;el('webcamStatus').className='webcam-status '+kind;};
  function stop(message='Камера выключена'){
    generation++;running=false;dialog.classList.remove('webcam-running');clearTimeout(timer);for(const c of controllers)c.abort();controllers.clear();
    stream?.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;
    worker?.terminate();worker=null;faceReady=false;facePending?.reject(Error('Остановлено'));facePending=null;
    evidence.reset();lastImage=null;completed=[];el('webcamSave').disabled=true;el('webcamStart').disabled=false;el('webcamStop').disabled=true;el('webcamDevice').disabled=false;el('webcamBackend').disabled=false;el('webcamRate').disabled=false;
    el('webcamPlaceholder').hidden=false;el('webcamPlaceholder').textContent='Камера выключена';
    ctx.clearRect(0,0,canvas.width,canvas.height);el('webcamMetrics').textContent='Обработка остановлена. Для продолжения нажмите «Запустить».';status(message);
  }
  function faceRequest(data,transfer=[]){return new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>{if(facePending){facePending=null;reject(Error('Истекло время определения лица'));}},30000);
    facePending={resolve:v=>{clearTimeout(timeout);resolve(v);},reject:e=>{clearTimeout(timeout);reject(e);}};worker.postMessage(data,transfer);
  });}
  async function initFace(){
    worker=new Worker('/webcam-face-worker.mjs');
    worker.onmessage=({data})=>{const p=facePending;facePending=null;if(data.error)p?.reject(Error(data.error));else p?.resolve(data);};
    worker.onerror=e=>{facePending?.reject(Error(e.message||'Ошибка модуля лица'));facePending=null;};
    await faceRequest({type:'init'});faceReady=true;
  }
  function drawBox(box,color,label){
    const [x1,y1,x2,y2]=box.map((v,i)=>v*(i%2?canvas.height:canvas.width));ctx.strokeStyle=color;ctx.lineWidth=3;ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    if(label){ctx.font='bold 17px system-ui';const w=ctx.measureText(label).width+12,y=Math.max(0,y1-26);ctx.fillStyle=color;ctx.fillRect(Math.min(x1,canvas.width-w),y,w,25);ctx.fillStyle='#10233b';ctx.fillText(label,Math.min(x1,canvas.width-w)+6,y+18);}
  }
  function dispatch(token,run){
    if(!running||token!==generation)return;
    timer=setTimeout(()=>dispatch(token,run),errors?1500:1000/targetRate);
    const limit=el('webcamBackend').value==='linq'?2:1;
    if(run.inflight+run.ordered.size>=limit||run.captureOwner||video.readyState<2||!video.videoWidth)return;
    run.inflight++;run.captureOwner=++run.seq;
    tick(token,run,run.seq).finally(()=>{run.inflight--;});
  }
  function deliver(token,run,id,value){
    if(token!==generation)return;
    for(const item of run.ordered.complete(id,value)){
      if(token!==generation)return;
      if(item.error){
        evidence.reset();lastImage=null;el('webcamSave').disabled=true;ctx.clearRect(0,0,canvas.width,canvas.height);
        errors++;status(`Обработка недоступна: ${item.error.name==='AbortError'?'сервер не ответил вовремя':item.error.message}. ${errors<3?'Повтор подключения…':'Нажмите «Запустить».'}`,'unknown');
        if(errors>=3){stop(el('webcamStatus').textContent);return;}
        continue;
      }
      const {result,crop,snapshot,geometry,encoded,captured,started}=item;
      const detections=mapDetections(result.prediction,crop,snapshot.width,snapshot.height),matches=mouthMatches(detections,geometry.mouth);
      const confirmed=evidence.observe(matches.length>0,geometry.face,started);
      canvas.width=snapshot.width;canvas.height=snapshot.height;ctx.drawImage(snapshot,0,0);
      if(geometry.mouth)drawBox(geometry.mouth,'#4ed8f3','Губы');
      for(const d of detections.filter(d=>d.class_name!=='person'))drawBox(d.xyxy,matches.includes(d)?'#ff697b':'#f9c653',`${d.class_name} ${Math.round(d.confidence*100)}%`);
      lastImage=confirmed?encoded:null;el('webcamSave').disabled=!confirmed||saveBusy;
      if(confirmed)status('Сигарета у рта · совпадение на 3+ кадрах','alert');
      else if(matches.length)status(`Сигарета возле губ · проверка ${evidence.count}/3`,'candidate');
      else if(!geometry.mouth)status(faceUnavailable?'Контур губ недоступен · только детекция объектов':'Лицо не найдено · приблизьтесь к камере','unknown');
      else if(detections.some(d=>d.class_name==='cigarette'))status('Сигарета обнаружена · совпадения с областью губ нет','candidate');
      else status('Сигарета у рта не обнаружена в этом кадре');
      const elapsed=performance.now()-started;
      completed.push(performance.now());if(completed.length>12)completed.shift();const fps=completed.length>1?(completed.length-1)*1000/(completed.at(-1)-completed[0]):0;
      el('webcamMetrics').textContent=`Кадр ${captured} · inference ${Math.round(result.prediction.inference_ms)} мс · полный цикл ${Math.round(elapsed)} мс · ${fps.toFixed(1)} кадров/с · ${result.prediction.backend}${result.prediction.preset?' / '+result.prediction.preset:''} · ${result.prediction.epochs} эпох`;
      errors=0;
    }
  }
  async function tick(token,run,id){
    if(!running||token!==generation)return;
    const started=performance.now(),snapshot=document.createElement('canvas'),cropCanvas=document.createElement('canvas');let timeout,controller;
    try{
      const scale=Math.min(1,1280/video.videoWidth,960/video.videoHeight);
      snapshot.width=Math.round(video.videoWidth*scale);snapshot.height=Math.round(video.videoHeight*scale);
      snapshot.getContext('2d').drawImage(video,0,0,snapshot.width,snapshot.height);
      const captured=new Date().toLocaleTimeString('ru-RU');
      let geometry={face:null,mouth:null};
      if(faceReady){const bitmap=await createImageBitmap(snapshot);geometry=await faceRequest({type:'detect',bitmap},[bitmap]);}
      if(token!==generation)return;
      // Crop around the face to retain a tiny cigarette at the model's 640px input.
      let crop={x:0,y:0,w:snapshot.width,h:snapshot.height};
      if(geometry.face){const [x1,y1,x2,y2]=geometry.face,w=x2-x1,h=y2-y1;
        const l=Math.max(0,x1-w*.4),t=Math.max(0,y1-h*.2),r=Math.min(1,x2+w*.4),b=Math.min(1,y2+h*.25);
        crop={x:Math.floor(l*snapshot.width),y:Math.floor(t*snapshot.height),w:Math.max(1,Math.round((r-l)*snapshot.width)),h:Math.max(1,Math.round((b-t)*snapshot.height))};
      }
      cropCanvas.width=crop.w;cropCanvas.height=crop.h;cropCanvas.getContext('2d').drawImage(snapshot,crop.x,crop.y,crop.w,crop.h,0,0,crop.w,crop.h);
      const encoded=cropCanvas.toDataURL('image/jpeg',.9).split(',')[1];
      if(run.captureOwner===id)run.captureOwner=0;controller=new AbortController();controllers.add(controller);timeout=setTimeout(()=>controller.abort(),65000);
      const response=await fetch(el('webcamBackend').value==='linq'?'/api/demo/webcam/linq':'/api/demo/webcam',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:encoded}),signal:controller.signal});
      clearTimeout(timeout);
      if(token!==generation)return;
      if(!response.ok){let message;try{message=(await response.json()).error;}catch{}throw Error(message||`Сервер: HTTP ${response.status}`);}
      const result=await response.json();if(token!==generation)return;
      if(el('webcamBackend').value==='linq'&&result.prediction.backend!=='linq_amqp')throw Error('Вместо LinQ получен другой backend');
      deliver(token,run,id,{result,crop,snapshot,geometry,encoded,captured,started});
    }catch(e){
      if(run.captureOwner===id)run.captureOwner=0;deliver(token,run,id,{error:e});
    }finally{
      clearTimeout(timeout);if(controller)controllers.delete(controller);
    }
  }
  async function start(){
    stop();const token=generation;targetRate=Math.min(el('webcamBackend').value==='linq'?10:2,Number(el('webcamRate').value));el('webcamStart').disabled=true;el('webcamStop').disabled=false;el('webcamDevice').disabled=true;el('webcamBackend').disabled=true;el('webcamRate').disabled=true;status('Запрос доступа к камере…');errors=0;faceUnavailable=false;
    try{
      if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia)throw Error('Камера доступна через HTTPS или localhost. Откройте защищённый адрес сайта.');
      const device=el('webcamDevice').value;
      const acquired=await navigator.mediaDevices.getUserMedia({audio:false,video:{width:{ideal:1280},height:{ideal:720},...(device?{deviceId:{exact:device}}:{facingMode:'user'})}});
      if(token!==generation){acquired.getTracks().forEach(t=>t.stop());return;}
      stream=acquired;stream.getVideoTracks()[0].onended=()=>{if(running)stop('Камера отключена. Подключите её и повторите запуск.');};
      video.srcObject=stream;await video.play();if(token!==generation)return;
      el('webcamPlaceholder').hidden=true;
      const devices=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
      if(token!==generation)return;
      el('webcamDevice').replaceChildren(new Option('Камера по умолчанию',''),...devices.map((d,i)=>new Option(d.label||`Камера ${i+1}`,d.deviceId)));
      el('webcamDevice').value=stream.getVideoTracks()[0].getSettings().deviceId||device;
      status('Загрузка контура лица и модели…');
      try{await initFace();}catch(e){if(token!==generation)return;faceUnavailable=true;worker?.terminate();worker=null;faceReady=false;}
      if(token!==generation)return;running=true;dialog.classList.add('webcam-running');dispatch(token,{seq:0,inflight:0,captureOwner:0,ordered:new OrderedFrames()});
    }catch(e){if(token===generation)stop(cameraError(e));}
  }
  el('webcamStart').onclick=start;el('webcamStop').onclick=()=>stop();
  el('webcamBackend').onchange=()=>{const linq=el('webcamBackend').value==='linq';for(const o of el('webcamRate').options)o.disabled=!linq&&Number(o.value)>2;el('webcamRate').value=linq?'5':'2';el('webcamBackendLabel').textContent=`YOLOv8n · 26 эпох · ${linq?'LinQ TPU через Jupyter':'CPU сервера'} + контур губ в браузере`;};
  fetch('/api/demo/linq-health',{signal:AbortSignal.timeout(5000)}).then(r=>r.ok?r.json():null).then(data=>{if(data?.ready&&data.backend==='linq_amqp'&&dialog.isConnected){const option=el('webcamBackend').querySelector('[value=linq]');option.disabled=false;option.textContent='LinQ TPU · '+data.preset;if(!stream&&!el('webcamStart').disabled){el('webcamBackend').value='linq';el('webcamBackend').onchange();}}else if(dialog.isConnected){el('webcamBackend').querySelector('[value=linq]').textContent='LinQ TPU · недоступен';}}).catch(()=>{if(dialog.isConnected)el('webcamBackend').querySelector('[value=linq]').textContent='LinQ TPU · недоступен';});
  el('webcamClose').onclick=()=>dialog.close();dialog.addEventListener('close',()=>{stop();current=null;dialog.remove();});
  const visibility=()=>{if(document.hidden&&stream)stop('Вкладка скрыта — камера и обработка остановлены.');};
  document.addEventListener('visibilitychange',visibility);dialog.addEventListener('close',()=>document.removeEventListener('visibilitychange',visibility),{once:true});
  const pagehide=()=>stop();window.addEventListener('pagehide',pagehide);
  dialog.addEventListener('close',()=>window.removeEventListener('pagehide',pagehide),{once:true});
  el('webcamSave').onclick=async()=>{
    if(!lastImage||saveBusy)return;const image=lastImage;saveBusy=true;stop('Камера остановлена · сохранение срабатывания…');
    try{let r;for(let attempt=0;attempt<5;attempt++){
        r=await fetch('/api/demo/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image,source:'webcam'})});
        if(r.status!==409)break;await new Promise(resolve=>setTimeout(resolve,750));
      }const data=await r.json();if(!r.ok)throw Error(data.error);
      if(typeof window.refresh==='function')await window.refresh();
      status(data.event?'Кадр сохранён в журнал · решение принимает оператор':data.message,'candidate');
    }catch(e){status('Не удалось сохранить: '+e.message,'unknown');}finally{saveBusy=false;el('webcamSave').disabled=!lastImage;}
  };
  current={dialog};
}
