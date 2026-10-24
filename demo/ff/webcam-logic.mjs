// Geometry is evaluated on the same captured frame as YOLO, never on a later video frame.
export function intersects(a,b){return a[0]<b[2]&&a[2]>b[0]&&a[1]<b[3]&&a[3]>b[1];}
export function iou(a,b){const w=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0])),h=Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));return w*h/Math.max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-w*h);}
export function mapDetections(prediction,crop,width,height){return prediction.detections.map(d=>({...d,xyxy:[crop.x+d.xyxy[0]/prediction.size[0]*crop.w,crop.y+d.xyxy[1]/prediction.size[1]*crop.h,crop.x+d.xyxy[2]/prediction.size[0]*crop.w,crop.y+d.xyxy[3]/prediction.size[1]*crop.h].map((x,i)=>x/(i%2?height:width))}));}
export function mouthMatches(detections,mouth){
  if(!mouth)return [];
  return detections.filter(d=>d.class_name==='cigarette'&&d.confidence>=.25&&intersects(d.xyxy,mouth)&&(d.xyxy[2]-d.xyxy[0])<=4*(mouth[2]-mouth[0])&&(d.xyxy[3]-d.xyxy[1])<=4*(mouth[3]-mouth[1]));
}
export class Evidence{
  constructor(){this.reset();}
  reset(){this.count=0;this.start=0;this.last=0;this.face=null;}
  observe(matched,face,now){
    if(!matched||!face){this.reset();return false;}
    if(!this.count||now-this.last>5000||(this.face&&iou(face,this.face)<.1)){this.reset();this.start=now;}
    this.count++;this.last=now;this.face=face;
    return this.count>=3&&now-this.start>=500;
  }
}
// At most two captured frames are admitted by the caller. Completion order may differ.
export class OrderedFrames{
  constructor(){this.next=1;this.pending=new Map();}
  complete(id,value){
    if(id<this.next)return [];
    this.pending.set(id,value);const ready=[];
    while(this.pending.has(this.next)){ready.push(this.pending.get(this.next));this.pending.delete(this.next++);}
    return ready;
  }
  get size(){return this.pending.size;}
}
