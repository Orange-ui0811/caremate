const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../..'),out=process.env.VERIFICATION_DIR;
const sse=events=>events.map(e=>'data: '+JSON.stringify(e)+'\n\n').join('')+'data: [DONE]\n\n';
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 try{
  fs.mkdirSync(out,{recursive:true});const page=await browser.newPage(),errors=[];let submitted=0;
  page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{localStorage.setItem('mewhelp_session_id','alice');localStorage.setItem('caremate_multi_intent_mode','multi');});
  await page.route('**/*',async route=>{
   const req=route.request(),u=new URL(req.url());
   if(u.hostname!=='mewhelp.test')return route.fulfill({body:''});
   if(u.pathname==='/api/conversations')return route.fulfill({json:{items:[]}});
   if(u.pathname.endsWith('/workflow-mode'))return route.fulfill({json:{multi_intent_enabled:true}});
   if(u.pathname.endsWith('/messages'))return route.fulfill({json:{items:[]}});
   if(u.pathname.endsWith('/task-events'))return route.fulfill({json:{items:[],interrupt:null}});
   if(u.pathname.endsWith('/ack'))return route.fulfill({json:{ok:true}});
   if(u.pathname==='/api/chat')return route.fulfill({contentType:'text/event-stream',body:sse([
    {event:'task_start'},{event:'citations',items:[{n:1,id:15,answer:'外套可装洗衣袋冷水机洗'}]},
    {delta:'外套可机洗[1]。'},
    {event:'task_start'},{event:'tool',name:'query_order'},
    {event:'task_event',conversation_id:7,event_id:'t:pending',message_id:10,text:'订单1001的退款申请尚未提交，请在下方卡片确认。'},
    {event:'interrupt',kind:'task_form',conversation_id:7,interrupt_id:'form',request_id:'r',form:{type:'refund_form',draft:{order_id:'1001',reason:'质量问题'}}}
   ])});
   if(u.pathname==='/api/actions/create-refund'){submitted++;return route.fulfill({json:{ticket_no:'T1',status:'退款申请已提交'}});}
   if(u.pathname==='/api/actions/resume')return route.fulfill({contentType:'text/event-stream',body:sse([
    {event:'task_event',conversation_id:7,event_id:'t:done',message_id:11,text:'退款申请已提交，工单号：T1。仍需平台审核。'},
    {event:'task_start'},{delta:'另一笔订单物流已揽件，成都。'},
    {event:'done',conversation_id:7}
   ])});
   const file=path.resolve(root,'app/static',u.pathname==='/'?'index.html':u.pathname.replace('/static/',''));
   assert(file.startsWith(path.join(root,'app/static')+path.sep));
   return fs.existsSync(file)?route.fulfill({body:fs.readFileSync(file),contentType:{'.html':'text/html','.css':'text/css','.js':'application/javascript'}[path.extname(file)]||'application/octet-stream'}):route.fulfill({status:404,body:''});
  });
  await page.goto('https://mewhelp.test/');const idle=()=>page.waitForFunction(()=>!document.getElementById('send').disabled);
  await idle();await page.locator('#input').fill('查说明，申请退款，再查另一单物流');await page.locator('#send').click();await idle();
  assert.match(await page.locator('#messages').innerText(),/尚未提交/);
  await page.locator('.cite').click();assert.match(await page.locator('.cite-pop').innerText(),/洗衣袋/);await page.keyboard.press('Escape');
  await page.getByRole('button',{name:'提交退款工单',exact:true}).click();
  await page.locator('#refundReason').selectOption({label:'质量问题'});await page.locator('#refundSubmit').click();
  await page.waitForFunction(()=>document.getElementById('messages').innerText.includes('另一笔订单物流'));await idle();
  const text=await page.locator('#messages').innerText();
  assert.equal(submitted,1);assert.equal((text.match(/工单号：T1/g)||[]).length,1);
  assert(!text.includes('(无回复)'));assert(!text.includes('task_id'));assert.deepEqual(errors,[]);
  await page.screenshot({path:path.join(out,'completion-events.png')});
  fs.writeFileSync(path.join(out,'result.json'),JSON.stringify({pass:true,submitted,errors}));
  console.log('PASS mixed streaming/status events, original refund form, continuation, citations, no blank duplicate replies');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
