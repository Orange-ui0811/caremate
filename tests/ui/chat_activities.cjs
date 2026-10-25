const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '../..');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1200, height: 820 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    const activities = [{ kind: 'retrieval', name: 'retrieve_knowledge' }, { kind: 'tool', name: 'query_product' }];
    const events = [{ event: 'activity', ...activities[0] }, { event: 'tool', name: 'query_product' }, { delta: '已找到退换货政策。' }, { event: 'done', conversation_id: 77 }];
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/api/chat') return route.fulfill({ contentType: 'text/event-stream', body: events.map(x => 'data: ' + JSON.stringify(x) + '\n\n').join('') + 'data: [DONE]\n\n' });
      if (url.pathname === '/api/conversations') return route.fulfill({ json: { items: [{ id: 77, preview: '退货换货政策是怎么样的' }] } });
      if (url.pathname.endsWith('/messages')) return route.fulfill({ json: { items: [
        { role: 'user', content: '退货换货政策是怎么样的' },
        { role: 'assistant', content: '已找到退换货政策。', activities }
      ] } });
      const file = url.pathname === '/' ? 'index.html' : url.pathname.replace('/static/', '');
      const types = { '.html': 'text/html', '.css': 'text/css', '.js': 'application/javascript', '.svg': 'image/svg+xml' };
      return route.fulfill({ body: fs.readFileSync(path.join(root, 'app/static', file)), contentType: types[path.extname(file)] });
    });
    await page.goto('http://localhost:8000/');
    await page.locator('#input').fill('退货换货政策是怎么样的');
    await page.locator('#send').click();
    await page.waitForFunction(() => !document.getElementById('send').disabled);
    assert.deepEqual(await page.locator('.tool-badge').allTextContents(), ['已检索知识库', '已调用 query_product']);
    for (const badge of await page.locator('.tool-badge').all()) assert(await badge.isVisible());
    fs.mkdirSync(path.join(root, 'data/verification/activities-20261012'), { recursive: true });
    await page.screenshot({ path: path.join(root, 'data/verification/activities-20261012/live.png'), animations: 'disabled' });
    await page.reload();
    await page.locator('.conv-item').click();
    await page.waitForFunction(() => !document.getElementById('send').disabled);
    assert.deepEqual(await page.locator('.tool-badge').allTextContents(), ['已检索知识库', '已调用 query_product']);
    await page.setViewportSize({ width: 390, height: 844 });
    for (const badge of await page.locator('.tool-badge').all()) assert(await badge.isVisible());
    await page.screenshot({ path: path.join(root, 'data/verification/activities-20261012/mobile.png'), animations: 'disabled' });
    assert.deepEqual(errors, []);
    console.log('PASS: live retrieval/tool badges, history reload, desktop/mobile visibility, no JS errors.');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
