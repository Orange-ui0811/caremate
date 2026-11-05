// Real page and original forms; API receipts are isolated browser fixtures.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../..');
const out=process.env.VERIFICATION_DIR;assert(out);fs.mkdirSync(out,{recursive:true});
const sse=events=>events.map(e=>'data: '+JSON.stringify(e)+'\n\n').join('')+'data: [DONE]\n\n';
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 try {
  const page=await browser.newPage({viewport:{width:1200,height:850}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{localStorage.setItem('mewhelp_session_id','alice');localStorage.setItem('caremate_multi_intent_mode','multi');});
  let submitted=0,resumed=0,chats=0;
  await page.route('**/*',async route=>{
   const req=route.request(),url=new URL(req.url());
   if(url.hostname!=='mewhelp.test')return route.fulfill({body:''});
   if(url.pathname==='/api/conversations')return route.fulfill({json:{items:[]}});
   if(url.pathname.endsWith('/workflow-mode'))return route.fulfill({json:{multi_intent_enabled:true}});
   if(url.pathname.endsWith('/messages')||url.pathname.endsWith('/task-events'))return route.fulfill({json:{items:[],interrupt:null}});
   if(url.pathname==='/api/chat'){
    chats++;assert.equal(req.postDataJSON().multi_intent_enabled,true);
    return route.fulfill({contentType:'text/event-stream',body:sse([
     {event:'task_start'},{delta:'**商品信息**\n- 智能猫砂盆。'},
     {event:'task_start'},{delta:'**物流信息**\n- 已揽件，成都。'},
     {event:'task_start'},{delta:'请在原退款表单确认。'},
     {event:'interrupt',kind:'task_form',conversation_id:7,interrupt_id:'form-'+chats,request_id:'refund-'+chats,
      form:{type:'refund_form',draft:{order_id:'1001',reason:'质量问题'}}}
    ])});
   }
   if(url.pathname==='/api/actions/create-refund'){
    const body=req.postDataJSON();submitted++;assert.equal(body.order_id,'1001');assert.equal(body.request_id,'refund-1');
    return route.fulfill({json:{ticket_no:'T-refund',status:'退款申请已提交'}});
   }
   if(url.pathname==='/api/actions/resume'){
    const body=req.postDataJSON();resumed++;assert.equal(body.conversation_id,7);
    if(resumed===1){assert.equal(submitted,1);assert.equal(body.interrupt_id,'form-1');assert.equal(body.confirmed,true);
     return route.fulfill({contentType:'text/event-stream',body:sse([
      {event:'task_start'},{delta:'请确认投诉工单内容。'},
      {event:'interrupt',kind:'confirm_ticket',conversation_id:7,interrupt_id:'ticket-1',preview:{ticket_type:'投诉',description:'商品无法开机'}}
     ])});}
    if(resumed===2){assert.equal(body.interrupt_id,'ticket-1');assert.equal(body.confirmed,true);}
    else {assert.equal(body.interrupt_id,'form-2');assert.equal(body.confirmed,false);assert.equal(submitted,1);}
    return route.fulfill({contentType:'text/event-stream',body:sse([
      {delta:resumed===2?'投诉工单已提交，工单号 T-complaint。':'已取消退款，继续处理投诉。'},
      {event:'done',conversation_id:7}
    ])});
   }
   const file=path.resolve(root,'app/static',url.pathname==='/'?'index.html':url.pathname.replace('/static/',''));
   assert(file.startsWith(path.join(root,'app/static')+path.sep));
   return fs.existsSync(file)?route.fulfill({body:fs.readFileSync(file),contentType:{'.html':'text/html','.css':'text/css','.js':'application/javascript'}[path.extname(file)]||'application/octet-stream'}):route.fulfill({status:404,body:''});
  });
  await page.goto('https://mewhelp.test/');
  const idle=()=>page.waitForFunction(()=>!document.getElementById('send').disabled);
  async function ask(){await page.locator('#input').fill('查询商品物流，申请退款并登记投诉');await page.locator('#send').click();await idle();}
  await ask();
  assert.equal(await page.locator('#messages strong').filter({hasText:'商品信息'}).count(),1);
  assert.equal(await page.locator('#messages strong').filter({hasText:'物流信息'}).count(),1);
  await page.getByRole('button',{name:'提交退款工单',exact:true}).click();
  assert.equal(await page.locator('#refundOrder').inputValue(),'1001');
  await page.locator('#refundReason').selectOption({label:'质量问题'});
  await page.locator('#refundSubmit').click();
  await page.waitForSelector('.ticket-confirm');await idle();
  await page.waitForFunction(()=>getComputedStyle(document.getElementById('refundMask')).visibility==='hidden');
  await page.screenshot({path:path.join(out,'shared-agent-confirmation.png')});
  await page.getByRole('button',{name:'确认提交',exact:true}).click();await idle();
  assert.equal(resumed,2);assert.equal(await page.locator('.workflow-controls').count(),0);
  await ask();await page.getByRole('button',{name:'取消本项，继续其他事项',exact:true}).click();await idle();
  assert.equal(resumed,3);assert.equal(submitted,1);assert.deepEqual(errors,[]);
  assert(!/task_id|operation_id|本次事项处理结果/.test(await page.locator('#messages').innerText()));
  await page.screenshot({path:path.join(out,'shared-agent-completed.png')});
  fs.writeFileSync(path.join(out,'ui-result.json'),JSON.stringify({pass:true,submitted,resumed,errors},null,2));
  console.log('PASS original forms, streamed Markdown bubbles, resume, cancellation, no internal status');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
