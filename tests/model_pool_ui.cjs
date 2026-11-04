// Isolated browser fixtures: no real keys, model calls or production writes.
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');
let managed = false, submitted;
const requests = [], errors = [];
const runtime = () => ({
  enabled:true, policy:{global_limit:80,key_limit:8,background_limit:40,max_queue:256,queue_timeout:30,call_timeout:90,read_timeout:60,max_attempts:2},
  turn_timeout:180,replay_ttl:86400,running:[],waiting:0,replay_counts:{},
  recent_calls:[
    {request_id:'turn-1',channel:'chat',purpose:'coreference',purpose_label:'指代消解',priority:'foreground'},
    {request_id:'turn-1',channel:'chat',purpose:'query_expansion',purpose_label:'多路查询扩写',priority:'foreground'},
    {request_id:'legacy',channel:'chat',priority:'background'},
  ].map(row=>({model:'test',resource:'fp',attempt:1,queue_ms:0,started:1,finished:2,outcome:'SUCCEEDED',...row})),
  slots:managed?[{resource:'fingerprint',channels:['chat'],managed:true}]:[],
});
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
    await page.route('**/*',async route=>{
      const req=route.request(),url=new URL(req.url());requests.push([url.pathname,req.method()]);
      if(url.pathname==='/api/model-pool/keys' && req.method()==='POST') {
        submitted=req.postDataJSON();managed=true;return route.fulfill({json:{resources:['fingerprint']}});
      }
      if(url.pathname==='/api/model-pool/keys/fingerprint' && req.method()==='DELETE') {
        managed=false;return route.fulfill({json:{saved:true}});
      }
      if(url.pathname==='/api/model-pool/runtime')return route.fulfill({json:runtime()});
      if(url.pathname==='/static/admin.js')return route.fulfill({contentType:'application/javascript',body:'function mountAdminNav() {}'});
      const file=url.pathname==='/model-pool'?'model-pool.html':path.basename(url.pathname);
      if(['model-pool.html','model-pool.js','apple-theme.css','brand.svg'].includes(file)) {
        const contentType=file.endsWith('.js')?'application/javascript':file.endsWith('.css')?'text/css':file.endsWith('.svg')?'image/svg+xml':'text/html';
        return route.fulfill({contentType,body:fs.readFileSync(path.join(root,'app/static',file))});
      }
      return route.abort();
    });
    await page.goto('http://pool.test/model-pool');
    await page.getByRole('cell',{name:'指代消解',exact:false}).waitFor();
    assert((await page.locator('#calls').textContent()).includes('多路查询扩写'));
    assert((await page.locator('#calls').textContent()).includes('历史调用，未记录环节'));
    assert((await page.locator('#calls').textContent()).includes('聊天 · 前台'));
    assert.equal(await page.getByRole('columnheader',{name:'客服环节'}).count(),1);
    assert.equal(await page.locator('[name="key_limit"]').getAttribute('max'),'128');
    await page.locator('#new-keys').fill('test-secret-one\ntest-secret-two');
    await page.locator('#add-keys').click();
    await page.getByRole('button',{name:'移除',exact:true}).waitFor();
    assert.deepEqual(submitted.keys,['test-secret-one','test-secret-two']);
    assert.equal(await page.locator('#new-keys').inputValue(),'');
    assert(!(await page.locator('#slots').textContent()).includes('test-secret-one'));
    await page.getByRole('button',{name:'移除',exact:true}).click();
    await page.waitForFunction(()=>!document.querySelector('#slots')?.textContent.includes('fingerprint'));
    assert(requests.some(([url,method])=>url.endsWith('/keys/fingerprint') && method==='DELETE'));
    assert.deepEqual(errors,[]);
    await page.screenshot({path:path.join(root,'data/verification/model-pool-purpose-ui-20261026.png'),fullPage:true});
    console.log('Model pool purpose labels, legacy history and key form: passed');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
