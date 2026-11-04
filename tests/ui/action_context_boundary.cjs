// All HTTP requests are fulfilled locally; no application/API/model is contacted.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '../..');
const sse = events => events.map(e => 'data: ' + JSON.stringify(e) + '\n\n').join('') + 'data: [DONE]\n\n';
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage();
    const errors = [], requests = [];
    let chats = 0;
    page.on('pageerror', e => errors.push(e.message));
    await page.addInitScript(() => localStorage.setItem('mewhelp_session_id', 'alice'));
    await page.route('**/*', async route => {
      const req = route.request(), url = new URL(req.url());
      if (url.hostname !== 'mewhelp.test') return route.fulfill({ body: '' });
      if (url.pathname === '/api/conversations') return route.fulfill({ json: { items: [] } });
      if (url.pathname === '/api/chat') {
        chats++;
        const events = chats === 1 ? [{ event: 'interrupt', kind: 'confirm_ticket', conversation_id: 7,
          interrupt_id: 'ticket-interrupt', preview: { description: '合成咨询', ticket_type: '咨询' } }]
          : chats === 2 ? [{ event: 'interrupt', kind: 'select_order', conversation_id: 7,
            interrupt_id: 'order-interrupt', orders: [{ order_id: '1001', product: '合成商品', amount: 10 }] }]
          : [{ delta: '请使用下方表单。' }, { event: 'actions', items: [{ type: 'create_ticket' },
              { type: 'refund_form', draft: { order_id: '1001' } }] }, { event: 'done', conversation_id: 7 }];
        return route.fulfill({ body: sse(events), contentType: 'text/event-stream' });
      }
      if (url.pathname.startsWith('/api/actions/')) {
        const body = req.postDataJSON();
        assert.equal(body.user_id, 'alice');
        assert.equal(body.conversation_id, 7);
        requests.push({ path: url.pathname, body });
        if (url.pathname.endsWith('/resume')) return route.fulfill({ body: sse([
          { delta: '合成确认已处理。' }, { event: 'done', conversation_id: 7 }]), contentType: 'text/event-stream' });
        return route.fulfill({ json: { ticket_no: 'SYNTHETIC-' + requests.length } });
      }
      const relative = url.pathname === '/' ? 'index.html' : url.pathname.replace('/static/', '');
      const file = path.resolve(root, 'app/static', relative);
      assert.ok(file.startsWith(path.resolve(root, 'app/static') + path.sep));
      if (!fs.existsSync(file)) return route.fulfill({ status: 404, body: '' });
      const types = { '.html': 'text/html', '.css': 'text/css', '.js': 'application/javascript', '.svg': 'image/svg+xml' };
      return route.fulfill({ body: fs.readFileSync(file), contentType: types[path.extname(file)] || 'application/octet-stream' });
    });
    await page.goto('https://mewhelp.test/');
    async function ask(text) {
      await page.locator('#input').fill(text);
      await page.locator('#send').click();
      await page.waitForFunction(() => !document.getElementById('send').disabled);
    }
    await ask('合成工单确认');
    await page.getByRole('button', { name: '确认提交', exact: true }).click();
    await page.waitForFunction(() => !document.getElementById('send').disabled);
    assert.equal(requests[0].body.interrupt_id, 'ticket-interrupt');
    assert.equal(requests[0].body.confirmed, true);
    await ask('合成订单选择');
    await page.locator('.order-card').last().click();
    await page.waitForFunction(() => !document.getElementById('send').disabled);
    assert.equal(requests[1].body.interrupt_id, 'order-interrupt');
    assert.equal(requests[1].body.order_id, '1001');
    await ask('合成表单');
    await page.getByRole('button', { name: '建工单', exact: true }).click();
    await page.locator('#ticketType').selectOption('咨询');
    await page.locator('#ticketDesc').fill('合成咨询');
    await page.locator('#ticketSubmit').click();
    await page.waitForFunction(() => !document.getElementById('ticketMask').classList.contains('open'));
    assert.equal(requests[2].path, '/api/actions/create-ticket');
    await page.getByRole('button', { name: '提交退款工单', exact: true }).click();
    await page.locator('#refundReason').selectOption('质量问题');
    await page.locator('#refundSubmit').click();
    await page.waitForFunction(() => !document.getElementById('refundMask').classList.contains('open'));
    assert.equal(requests[3].path, '/api/actions/create-refund');
    assert.equal(requests[3].body.order_id, '1001');
    assert.deepEqual(errors, []);
    console.log('PASS: four UI actions carry caller identity; resume cards carry their interrupt IDs; zero live HTTP.');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
