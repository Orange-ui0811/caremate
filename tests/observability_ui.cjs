// Synthetic DOM checks: no live services, saved settings or browser state are touched.
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const assert = require('node:assert/strict');
const {parseHTML} = require('linkedom');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'app/static/observability.html'), 'utf8');
const {window, document} = parseHTML(html);
let threshold = .4, absentReport = false, exactRates = true, writes = 0;
let recommendation = .5, replay = false;
const job = name => ({name, title:name, status:'idle'});
const report = () => ({
  cost:{present:false, job:job('cost-report')}, trend:{present:false, job:job('eval-flywheel')},
  calibration:{present:!absentReport, job:job('calibrate-confidence'), in_use:threshold,
    scoring_method:'max_rerank_v1',
    meta:{source:replay?'recorded_api_replay':'live_retrieval',
      limitation:'内部校准回放，不代表最终答案质量。'},
    distribution:{answerable:{n:2}, absent:{n:2}},
    recommended:{threshold:recommendation, pass_rate:.5, leak_rate:0, youden_j:.5,
      selection_rule_label:replay?'校准集应拒放行≤5%，优先减少可答误拒':'Youden J 最大：可答通过率减应拒放行率'},
    in_use_rates:exactRates ? {pass_rate:[.4,.9].filter(x=>x>=threshold).length/2,
      leak_rate:[.1,.4].filter(x=>x>=threshold).length/2} : null,
    scan:[{t:.25,pass_rate:1,leak_rate:.5},{t:.4,pass_rate:1,leak_rate:.5},
      {t:.5,pass_rate:.5,leak_rate:0},{t:.95,pass_rate:0,leak_rate:0}]}
});
const context = vm.createContext({window, document, console,
  setTimeout:()=>0, clearTimeout(){}, setInterval:()=>0, clearInterval(){}, mountAdminNav(){},
  fetch:async (url, opts) => {
    if (url.endsWith('/confidence-threshold')) {threshold=JSON.parse(opts.body).threshold; writes++;}
    return {ok:true, json:async()=>url.endsWith('/overview')?report():{threshold}};
  }
});
vm.runInContext(fs.readFileSync(path.join(root,'app/static/acceptance.js'),'utf8'), context);
vm.runInContext([...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(x=>x[1]).join('\n'), context);
const tick=()=>new Promise(r=>setImmediate(r));
function dispatch(node,type,fields={}) {const e=new window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,fields);node.dispatchEvent(e);}
const kpi = label => [...document.querySelectorAll('.kpi')].find(n=>n.querySelector('.k-label').textContent===label)?.querySelector('.k-val').textContent;
(async()=>{
  await tick();
  assert.equal(kpi('在用阈值'),'.4'.replace(/^\./,'0.'));
  assert.equal(kpi('当前可答通过率'),'100%'); assert.equal(kpi('当前应拒放行率'),'50%');
  const svg=document.querySelector('svg.chart');
  // Exercise the SVG-to-screen inverse transform at a scaled, offset viewport.
  svg.getScreenCTM=()=>({inverse:()=>({})});
  svg.createSVGPoint=()=>({matrixTransform(){return {x:(this.x-20)/.5,y:(this.y-30)/.5};}});
  const x=42+(.4-.05)/.9*664;
  for(const y of [16,135]){
    dispatch(svg,'pointermove',{clientX:20+x*.5,clientY:30+y*.5});
    assert.equal(svg.querySelector('.scan-hover').getAttribute('visibility'),'visible');
    const text=svg.querySelector('.scan-tooltip').textContent;
    assert(text.includes('扫描阈值 0.4')&&text.includes('100%')&&text.includes('50%'));
  }
  dispatch(svg,'pointerleave');assert.equal(svg.querySelector('.scan-hover').getAttribute('visibility'),'hidden');
  dispatch(svg,'focus');dispatch(svg,'keydown',{key:'ArrowRight'});
  assert(svg.querySelector('.scan-tooltip').textContent.includes('扫描阈值 0.5'));
  assert(svg.getAttribute('aria-label').includes('50%'));
  dispatch(svg,'keydown',{key:'Escape'});assert.equal(svg.querySelector('.scan-hover').getAttribute('visibility'),'hidden');
  const input=document.getElementById('confidenceThreshold');input.value='0.4001';input.reportValidity=()=>true;
  dispatch(document.querySelector('.threshold-settings'),'submit');await tick();
  assert.equal(writes,1);assert.equal(kpi('在用阈值'),'0.4001');
  assert.equal(kpi('当前可答通过率'),'50%');assert.equal(kpi('当前应拒放行率'),'0%');
  assert(document.getElementById('body').textContent.includes('Youden J 最大'));
  replay=true;recommendation=.2260993940477108;await vm.runInContext('load()',context);
  assert.equal(kpi('推荐阈值'),'0.2261');
  assert(document.getElementById('body').textContent.includes('真实API记录回放'));
  assert(document.getElementById('body').textContent.includes('校准集应拒放行≤5%'));
  assert(document.getElementById('body').textContent.includes('不代表最终答案质量'));
  const apply=[...document.querySelectorAll('button')].find(n=>n.textContent.startsWith('应用推荐值'));
  dispatch(apply,'click');await tick();assert.equal(threshold,recommendation);
  exactRates=false;await vm.runInContext('load()',context);
  assert.equal(kpi('当前可答通过率'),'—');assert(document.getElementById('body').textContent.includes('请重跑校准'));
  absentReport=true;await vm.runInContext('load()',context);assert.equal(document.querySelector('svg.chart'),null);
  assert(document.getElementById('confidenceThreshold'));
  console.log('Passed: both curve hover values, scaled SVG coordinates, keyboard/leave, current-rate KPI values, saved threshold refresh, unavailable and missing reports. Synthetic DOM only.');
})().catch(e=>{console.error(e);process.exitCode=1});
