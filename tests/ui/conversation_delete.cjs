const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '../..');
const out = path.join(root, 'data/verification/conversation-delete-20261012');
fs.mkdirSync(out, { recursive: true });
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 850 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    let items = [{ id: 2, preview: '当前对话', has_summary: false }, { id: 1, preview: '旧对话 <img src=x>', has_summary: false }];
    let fail = false, deletes = 0, messageReads = 0;
    await page.route('**/*', async route => {
      const req = route.request(), url = new URL(req.url());
      if (url.pathname === '/api/conversations') return route.fulfill({ json: { items } });
      if (req.method() === 'DELETE') {
        assert.equal(url.searchParams.get('user_id'), 'ui-delete-test');
        deletes++;
        if (fail) return route.fulfill({ status: 503, json: { detail: 'failed' } });
        items = items.filter(it => it.id !== Number(url.pathname.split('/').at(-1)));
        return route.fulfill({ json: { ok: true } });
      }
      if (url.pathname.endsWith('/messages')) {
        messageReads++;
        return route.fulfill({ json: { items: [{ role: 'user', content: '当前对话内容' }] } });
      }
      const file = url.pathname === '/' ? 'index.html' : url.pathname.replace('/static/', '');
      const types = { '.html': 'text/html', '.css': 'text/css', '.js': 'application/javascript', '.svg': 'image/svg+xml' };
      return route.fulfill({ body: fs.readFileSync(path.join(root, 'app/static', file)), contentType: types[path.extname(file)] });
    });
    await page.addInitScript(() => {
      if (!localStorage.getItem('mewhelp_session_id')) {
        localStorage.setItem('mewhelp_session_id', 'ui-delete-test');
        localStorage.setItem('mewhelp_conversation_id', '2');
      }
    });
    await page.goto('http://mewhelp.test/');
    await page.locator('.conv-item').first().click();
    await page.waitForFunction(() => !document.getElementById('send').disabled);
    await page.getByRole('button', { name: '删除会话 #1', exact: true }).click();
    await page.getByRole('dialog', { name: '删除这段对话？' }).waitFor({ state: 'visible' });
    assert.equal(await page.locator('#deleteConvPreview img').count(), 0);
    await page.screenshot({ path: path.join(out, 'desktop.png'), animations: 'disabled' });
    await page.getByRole('button', { name: '取消', exact: true }).last().click();
    assert.equal(deletes, 0);
    assert.equal(messageReads, 1);
    await page.getByRole('button', { name: '删除会话 #1', exact: true }).click();
    await page.getByRole('button', { name: '删除对话', exact: true }).click();
    await page.waitForFunction(() => document.querySelectorAll('.conv-row').length === 1 && !document.getElementById('send').disabled);
    assert.equal(await page.evaluate(() => localStorage.getItem('mewhelp_conversation_id')), '2');
    assert.equal(await page.locator('#messages .row').count(), 1);
    fail = true;
    await page.getByRole('button', { name: '删除会话 #2', exact: true }).click();
    await page.getByRole('button', { name: '删除对话', exact: true }).click();
    await page.waitForFunction(() => document.getElementById('deleteConvError').textContent.includes('失败'));
    assert.equal(await page.locator('.conv-row').count(), 1);
    assert.equal(await page.evaluate(() => localStorage.getItem('mewhelp_conversation_id')), '2');
    fail = false;
    await page.getByRole('button', { name: '删除对话', exact: true }).click();
    await page.waitForFunction(() => !document.querySelector('.conv-row') && !document.getElementById('send').disabled);
    assert.equal(await page.evaluate(() => localStorage.getItem('mewhelp_conversation_id')), null);
    assert.equal(await page.locator('#messages .empty').count(), 1);
    await page.reload();
    await page.locator('.sb-empty').waitFor();
    assert.equal(await page.locator('.conv-row').count(), 0);
    items = [{ id: 9, preview: '手机端删除测试' }];
    await page.setViewportSize({ width: 390, height: 844 });
    await page.reload();
    await page.getByRole('button', { name: '打开会话记录' }).click();
    await page.getByRole('button', { name: '删除会话 #9', exact: true }).click();
    await page.getByRole('dialog', { name: '删除这段对话？' }).waitFor({ state: 'visible' });
    await page.screenshot({ path: path.join(out, 'mobile.png'), animations: 'disabled' });
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !document.getElementById('deleteConvMask').classList.contains('open'));
    await page.evaluate(() => setBusy(true));
    assert.equal(await page.locator('.conv-delete').isDisabled(), true);
    await page.evaluate(() => setBusy(false));
    await page.getByRole('button', { name: '删除会话 #9', exact: true }).click();
    await page.getByRole('button', { name: '删除对话', exact: true }).click();
    await page.waitForFunction(() => !document.querySelector('.conv-row') && !document.getElementById('send').disabled);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    assert.deepEqual(errors, []);
    console.log('PASS: cancel, inactive/current deletion, failure/retry, persistence, mobile, Escape, busy controls, no JS errors.');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
