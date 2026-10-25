// Synthetic DOM interaction test; no browser or real credentials.
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const {parseHTML} = require('linkedom');
const {window, document} = parseHTML(fs.readFileSync('app/static/model-pool.html', 'utf8'));
let managed = false, submitted;
const calls = [];
const context = vm.createContext({window, document, console, setInterval(){}, mountAdminNav(){}, confirm:()=>true,
  fetch:async (url, options={}) => {
    calls.push([url, options.method || 'GET']);
    if (url.endsWith('/keys')) {submitted=JSON.parse(options.body); managed=true; return {ok:true,json:async()=>({resources:['fingerprint']})};}
    if (options.method==='DELETE') managed=false;
    return {ok:true,json:async()=>({enabled:true,policy:{global_limit:80,key_limit:8,background_limit:40,max_queue:256,queue_timeout:30,call_timeout:90,read_timeout:60,max_attempts:2},
      turn_timeout:180,replay_ttl:86400,running:[],waiting:0,replay_counts:{},recent_calls:[],
      slots:managed?[{resource:'fingerprint',channels:['chat'],managed:true}]:[]})};
  }});
vm.runInContext(fs.readFileSync('app/static/model-pool.js','utf8'),context);
const tick=()=>new Promise(r=>setImmediate(r));
(async()=>{
  await tick();
  assert.equal(Number(document.querySelector('[name="key_limit"]').max),128);
  document.getElementById('new-keys').value='secret-one\nsecret-two';
  await document.getElementById('key-form').onsubmit({preventDefault(){}});
  assert.deepEqual(submitted.keys,['secret-one','secret-two']);
  assert.equal(document.getElementById('new-keys').value,'');
  assert(document.getElementById('slots').textContent.includes('fingerprint'));
  assert(!document.getElementById('slots').textContent.includes('secret-one'));
  const remove=[...document.querySelectorAll('#slots button')].find(n=>n.textContent==='移除');
  await remove.onclick();
  assert(calls.some(([url,method])=>url.endsWith('/keys/fingerprint') && method==='DELETE'));
  console.log('Model pool key form: passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
