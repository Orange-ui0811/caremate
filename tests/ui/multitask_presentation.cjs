// Isolated browser regression: serve real UI files, fulfill all APIs locally.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../..');
const out=path.join(root,'data/verification/p3-ui');fs.mkdirSync(out,{recursive:true});
const sse=events=>events.map(e=>'data: '+JSON.stringify(e)+'\n\n').join('')+'data: [DONE]\n\n';
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 try {
  const page=await browser.newPage({viewport:{width:1200,height:850}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{localStorage.setItem('mewhelp_session_id','alice');localStorage.setItem('caremate_multi_intent_mode','multi');});
  let chats=0,resumes=0; const saved=[],messages=[];
  const reply=(id,text,citations)=>{const e={event:'task_event',event_id:'reply-'+id,message_id:id,sequence:id,text,conversation_id:7,citations};saved.push(e);messages.push({id,role:'assistant',content:text});return e;};
  const refund={operation_id:'op-refund',preview:{ticket_type:'退款',description:'退款审核申请 订单号=1001 用户原述：商品漏电，申请退款'}};
  const complaint={operation_id:'op-complaint',preview:{ticket_type:'投诉',description:'投诉商品漏电和商家处理问题'}};
  const card=(id,actions)=>({event:'interrupt',kind:'task_actions',conversation_id:7,interrupt_id:id,actions});
  await page.route('**/*',async route=>{
   const req=route.request(),url=new URL(req.url());
   if(url.hostname!=='mewhelp.test')return route.fulfill({body:''});
   if(url.pathname==='/api/conversations')return route.fulfill({json:{items:chats?[{id:7,preview:'合成多事项'}]:[]}});
   if(url.pathname.endsWith('/workflow-mode'))return route.fulfill({json:{multi_intent_enabled:true}});
   if(url.pathname.endsWith('/messages'))return route.fulfill({json:{items:messages}});
   if(url.pathname.endsWith('/task-events'))return route.fulfill({json:{items:saved,interrupt:null}});
   if(url.pathname.endsWith('/ack'))return route.fulfill({json:{ok:true}});
   if(url.pathname==='/api/chat'){
    assert.equal(req.postDataJSON().multi_intent_enabled,true);
    chats++;messages.push({id:chats,role:'user',content:req.postDataJSON().message});
    if(chats===1){const e=reply(101,'**订单 1001（猫砂盆）**\n- 物流状态：已揽件\n- 当前位置：成都\n\n退货需要核实商品状况[1]。',[{n:1,id:1,section_path:'退货政策',answer:'合成政策'}]);
     return route.fulfill({contentType:'text/event-stream',body:sse([e,e,{event:'task_event',internal:true,text:'task_id DONE'},{event:'done',conversation_id:7}])});}
    if(chats>=3)return route.fulfill({contentType:'text/event-stream',body:sse([
     reply(chats===3?105:107,'退款审核前，需要先确认对应订单。'),
     {event:'interrupt',kind:'select_order',conversation_id:7,interrupt_id:'i-select-'+chats,task_kind:'refund',
      can_cancel:true,selection_prompt:'请选择要申请退款审核的订单：',orders:[{order_id:'1001',product:'合成猫砂盆',amount:100,status:'已付款'}]}
    ])});
    return route.fulfill({contentType:'text/event-stream',body:sse([reply(102,'退款审核和投诉申请已准备，尚未提交。'),card('i-1',[refund,complaint])])});
   }
   if(url.pathname==='/api/actions/resume'){
    const body=req.postDataJSON();resumes++;assert.equal(body.conversation_id,7);
    if(resumes>=3){
     if(resumes===3){assert.equal(body.order_id,'1001');assert.equal(body.interrupt_id,'i-select-3');}
     else {assert.equal(body.cancelled,true);assert.equal(body.interrupt_id,'i-select-4');}
     return route.fulfill({contentType:'text/event-stream',body:sse([
      reply(resumes===3?106:108,resumes===3?'已确认订单1001。':'已取消这次选单。'),{event:'done',conversation_id:7}])});
    }
    if(resumes===1){assert.equal(body.operation_id,'op-refund');assert.equal(body.interrupt_id,'i-1');assert.equal(body.confirmed,true);
     return route.fulfill({contentType:'text/event-stream',body:sse([reply(103,'退款审核工单已提交，工单号：T100。'),card('i-2',[complaint])])});}
    assert.equal(body.operation_id,'op-complaint');assert.equal(body.interrupt_id,'i-2');assert.equal(body.confirmed,false);
    return route.fulfill({contentType:'text/event-stream',body:sse([reply(104,'已取消这次投诉申请，未创建工单。'),{event:'done',conversation_id:7}])});
   }
   const file=path.resolve(root,'app/static',url.pathname==='/'?'index.html':url.pathname.replace('/static/',''));
   assert(file.startsWith(path.join(root,'app/static')+path.sep));
   if(!fs.existsSync(file))return route.fulfill({status:404,body:''});
   return route.fulfill({body:fs.readFileSync(file),contentType:{'.html':'text/html','.css':'text/css','.js':'application/javascript','.svg':'image/svg+xml'}[path.extname(file)]||'application/octet-stream'});
  });
  await page.goto('https://mewhelp.test/');
  const idle=()=>page.waitForFunction(()=>!document.getElementById('send').disabled);
  async function ask(text){await page.locator('#input').fill(text);await page.locator('#send').click();await idle();}
  await ask('订单和物流查询');
  assert.equal(await page.locator('[data-message-id="101"]').count(),1);
  assert.equal(await page.locator('.cite').count(),1);
  await ask('申请退款并投诉');
  assert.equal(await page.locator('.workflow-controls').count(),1);
  assert.equal(await page.locator('.task-preview').count(),2);
  assert.equal(await page.locator('[data-message-id="102"] .workflow-controls').count(),1);
  await page.screenshot({path:path.join(out,'pending.png')});
  await page.setViewportSize({width:390,height:844});
  await page.waitForFunction(()=>getComputedStyle(document.getElementById('sidebar')).visibility==='hidden');
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
  await page.screenshot({path:path.join(out,'pending-mobile.png')});
  await page.setViewportSize({width:1200,height:850});
  await page.getByRole('button',{name:'确认提交退款工单',exact:true}).click();await idle();
  assert.equal(await page.locator('.workflow-controls').count(),1);
  assert.equal(await page.locator('.task-preview').count(),1);
  assert.equal(await page.getByRole('button',{name:'确认提交退款工单',exact:true}).count(),0);
  await page.getByRole('button',{name:'取消投诉工单',exact:true}).click();await idle();
  assert.equal(await page.locator('.workflow-controls').count(),0);
  assert.equal(await page.locator('[data-message-id]').count(),4);
  await ask('选择订单后继续');
  assert.equal(await page.locator('[data-message-id="105"] .workflow-controls').count(),1);
  await page.locator('.order-card').click();await idle();
  assert.equal(await page.locator('.workflow-controls').count(),0);
  await ask('取消选单');
  await page.getByRole('button',{name:'取消这次选单',exact:true}).click();await idle();
  assert.equal(await page.locator('.workflow-controls').count(),0);
  await page.reload();await idle();
  assert.equal(await page.locator('[data-message-id]').count(),8);
  assert.equal(await page.locator('.cite').count(),1);
  const text=await page.locator('#messages').innerText();
  assert(!/task_id|DONE|本次事项处理结果|无回复/.test(text));
  assert.equal((text.match(/当前位置：成都/g)||[]).length,1);
  await page.screenshot({path:path.join(out,'history.png')});
  await page.setViewportSize({width:390,height:844});
  await page.waitForFunction(()=>getComputedStyle(document.getElementById('sidebar')).visibility==='hidden');
  await page.screenshot({path:path.join(out,'mobile.png')});
  assert.deepEqual(errors,[]);
  const crypto=require('node:crypto');
  const code=Object.fromEntries(['app/static/index.html','app/static/chat.css'].map(p=>[p,crypto.createHash('sha256').update(fs.readFileSync(path.join(root,p))).digest('hex')]));
  fs.writeFileSync(path.join(out,'result.json'),JSON.stringify({passed:true,resumes,errors,reply_rows:8,citations:1,code},null,2));
  console.log('PASS: grouped replies, duplicate events ignored, independent cards, stale cards retired, history/citations/mobile, no JS errors.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
