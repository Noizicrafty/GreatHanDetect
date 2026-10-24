self.exports={};
importScripts('/assets/mediapipe/vision_bundle.js');
const {FaceLandmarker,FilesetResolver}=self.exports;
let model;
const clamp=x=>Math.max(0,Math.min(1,x));
self.onmessage=async({data})=>{
  try{
    if(data.type==='init'){
      const files=await FilesetResolver.forVisionTasks(new URL('./assets/mediapipe/wasm/',self.location.href).href);
      model=await FaceLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:new URL('./assets/mediapipe/face_landmarker.task',self.location.href).href,delegate:'CPU'},runningMode:'IMAGE',numFaces:1,minFaceDetectionConfidence:.5,minFacePresenceConfidence:.5});
      self.postMessage({ready:true});return;
    }
    let result;
    try{result=model.detect(data.bitmap);}finally{data.bitmap.close();}
    const pts=result.faceLandmarks[0];
    if(!pts){self.postMessage({mouth:null,face:null});return;}
    const xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);
    const face=[clamp(Math.min(...xs)),clamp(Math.min(...ys)),clamp(Math.max(...xs)),clamp(Math.max(...ys))];
    const ids=[...new Set(FaceLandmarker.FACE_LANDMARKS_LIPS.flatMap(e=>[e.start,e.end]))];
    const lips=ids.map(i=>pts[i]),lx=lips.map(p=>p.x),ly=lips.map(p=>p.y);
    const x1=Math.min(...lx),x2=Math.max(...lx),y1=Math.min(...ly),y2=Math.max(...ly);
    const padX=(x2-x1)*.15,padY=Math.max((y2-y1)*.35,(face[3]-face[1])*.025);
    self.postMessage({face,mouth:[clamp(x1-padX),clamp(y1-padY),clamp(x2+padX),clamp(y2+padY)]});
  }catch(e){self.postMessage({error:e.message||'Не удалось определить лицо'});}
};
