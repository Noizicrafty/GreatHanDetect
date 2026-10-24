import test from 'node:test';
import assert from 'node:assert/strict';
import {mouthMatches,mapDetections,Evidence,OrderedFrames} from '../demo/ff/webcam-logic.mjs';
const mouth=[.4,.4,.6,.5],face=[.2,.1,.8,.8];
const cigarette=(box,confidence=.8)=>({class_name:'cigarette',xyxy:box,confidence});
test('only a cigarette overlapping the lips is a candidate',()=>{
  assert.equal(mouthMatches([cigarette([.55,.44,.7,.48])],mouth).length,1);
  assert.equal(mouthMatches([cigarette([.7,.7,.8,.8])],mouth).length,0);
  assert.equal(mouthMatches([{...cigarette([.45,.4,.5,.45]),class_name:'smoke'}],mouth).length,0);
});
test('no face, low confidence and oversized box cannot confirm',()=>{
  assert.equal(mouthMatches([cigarette([.4,.4,.5,.5])],null).length,0);
  assert.equal(mouthMatches([cigarette([.4,.4,.5,.5],.24)],mouth).length,0);
  assert.equal(mouthMatches([cigarette([0,0,1,1])],mouth).length,0);
});
test('crop coordinates map back to the exact original frame',()=>{
  const mapped=mapDetections({size:[100,100],detections:[cigarette([10,20,30,40])]}, {x:100,y:50,w:200,h:100},400,200);
  assert.deepEqual(mapped[0].xyxy,[.3,.35,.4,.45]);
});
test('three consecutive observations spanning at least half a second',()=>{
  const e=new Evidence();assert.equal(e.observe(true,face,100),false);assert.equal(e.observe(true,face,400),false);assert.equal(e.observe(true,face,700),true);
});
test('rapid observations do not bypass the time condition',()=>{
  const e=new Evidence();for(const t of [100,200,300])assert.equal(e.observe(true,face,t),false);
});
test('negative, missing face, gap and changed face reset confirmation',()=>{
  const e=new Evidence();e.observe(true,face,0);e.observe(true,face,500);e.observe(false,face,1000);assert.equal(e.count,0);
  e.observe(true,face,1100);e.observe(true,null,1600);assert.equal(e.count,0);
  e.observe(true,face,1700);assert.equal(e.observe(true,face,8000),false);assert.equal(e.count,1);
  assert.equal(e.observe(true,[0,0,.1,.1],8500),false);assert.equal(e.count,1);
});
test('out-of-order results preserve capture order and release buffers',()=>{
 const o=new OrderedFrames();assert.deepEqual(o.complete(2,'second'),[]);assert.equal(o.size,1);
 assert.deepEqual(o.complete(1,'first'),['first','second']);assert.equal(o.size,0);assert.deepEqual(o.complete(1,'duplicate'),[]);
});
test('late negative frame is evaluated before subsequent positive frames',()=>{
 const o=new OrderedFrames(),e=new Evidence();e.observe(true,face,0);e.observe(true,face,200);
 assert.deepEqual(o.complete(2,{matched:true,time:800}),[]);
 const ready=o.complete(1,{matched:false,time:600});
 const confirmed=ready.map(x=>e.observe(x.matched,face,x.time));assert.deepEqual(confirmed,[false,false]);assert.equal(e.count,1);
});
