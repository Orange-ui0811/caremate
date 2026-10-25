// Browser regression checks; the chat fixture stays inside this isolated browser context.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { chromium } = require('C:/Users/l/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'data/frontend-checks');
const quick = process.argv.includes('--quick');
fs.mkdirSync(output, { recursive: true });
const routes = ['/', '/admin', '/kb', '/rag-eval', '/review', '/observability', '/topics', '/topics/questions', '/acceptance', '/acceptance/eval', '/acceptance/data', '/acceptance/errors'];
const errors = [], results = [];
for (const file of fs.readdirSync(path.join(root, 'app/static'))) {
  const source = fs.readFileSync(path.join(root, 'app/static', file), 'utf8');
  if (file.endsWith('.js')) new vm.Script(source, { filename: file });
  if (file.endsWith('.html')) for (const script of source.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) new vm.Script(script[1], { filename: file });
}
(async () => {
  const browser = await chromium.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
    const page = await context.newPage();
    page.on('pageerror', e => errors.push({ url: page.url(), message: e.message }));
    if (!quick) {
      await page.goto('file:///C:/Users/l/.codex/skills/apple-liquid-glass/reference.html');
      await page.screenshot({ path: path.join(output, 'skill-reference.png'), fullPage: true });
    }
    for (const [device, viewport] of (quick ? [] : [['desktop', { width: 1440, height: 1000 }], ['mobile', { width: 390, height: 844 }]])) {
      await page.setViewportSize(viewport);
      for (const route of routes) {
        await page.goto('http://localhost:8000' + route, { waitUntil: 'networkidle' });
        await page.locator('body').waitFor();
        const metrics = await page.evaluate(() => ({ width: innerWidth, contentWidth: document.documentElement.scrollWidth, background: getComputedStyle(document.body).backgroundColor, font: getComputedStyle(document.body).fontFamily, title: document.querySelector('h1')?.textContent, nav: !!document.querySelector('nav') }));
        results.push({ device, route, ...metrics });
        const name = route === '/' ? 'chat' : route.slice(1).replaceAll('/', '-');
        await page.screenshot({ path: path.join(output, `${name}-${device}.png`), fullPage: true });
      }
    }
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('http://localhost:8000/');
    await page.screenshot({ path: path.join(output, 'chat-tablet.png') });
    await page.goto('http://localhost:8000/kb', { waitUntil: 'networkidle' });
    await page.locator('#sampleBtn').click();
    await page.locator('#previewBtn').click();
    await page.locator('#previewBox table.tbl tr').nth(1).waitFor();
    results.push({ check: 'knowledge chunk preview', passed: true });

    // Mock only the isolated UI session. No model calls, orders or tickets are created.
    await page.route('**/api/conversations*', route => route.fulfill({ json: { items: [{ id: 'design-ui-check', preview: '上一条消息', message_count: 2 }], total: 1 } }));
    await page.route('**/api/conversations/*/messages', route => route.fulfill({ json: { items: [{ role: 'user', content: '历史问题' }, { role: 'assistant', content: '历史回答' }] } }));
    await page.route('**/api/chat', route => route.fulfill({ contentType: 'text/event-stream', body: [
      { event: 'tool', name: 'query_faq' },
      { delta: '**您好**，可以查看退换货政策。[1]' },
      { event: 'citations', items: [{ n: 1, questions: '退换货规则', answer: '请保留包装并查看售后规则。', section_path: '售后说明', chunk_id: 'check-only' }] },
      { event: 'actions', items: [{ type: 'create_ticket', label: '创建工单' }, { type: 'refund_form', label: '申请退款', draft: { order_id: 'UI-CHECK-ONLY' } }] },
      { event: 'done', conversation_id: 'design-ui-check' }
    ].map(item => 'data: ' + JSON.stringify(item) + '\n\n').join('') + 'data: [DONE]\n\n' }));
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto('http://localhost:8000/', { waitUntil: 'networkidle' });
    await page.locator('#input').fill('界面回归检查');
    await page.locator('#input').press('Enter');
    const ticket = page.locator('.action-btn', { hasText: '建工单' });
    await ticket.waitFor();
    assert.equal(await page.locator('.tool-badge').textContent(), '已调用 query_faq');
    assert.equal(await page.locator('.bubble-text strong').textContent(), '您好');
    await page.locator('.cite').focus();
    await page.keyboard.press('Enter');
    await page.locator('.cite-pop').waitFor();
    assert.match(await page.locator('.cite-pop').textContent(), /请保留包装/);
    await page.keyboard.press('Escape');
    await ticket.click();
    await page.locator('#ticketMask.open').waitFor();
    assert.equal(await page.locator('#chat-main').evaluate(el => el.inert), true);
    await page.locator('#ticketType').selectOption('售后');
    await page.locator('#ticketDesc').fill('只测试表单，不提交。');
    await page.screenshot({ path: path.join(output, 'ticket-dialog.png') });
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !document.querySelector('#ticketMask').classList.contains('open'));
    assert.equal(await ticket.evaluate(el => document.activeElement === el), true);
    const refund = page.locator('.action-btn', { hasText: '提交退款工单' });
    await refund.click();
    await page.locator('#refundMask.open').waitFor();
    await page.locator('#refundCancel').click();
    assert.equal(await refund.evaluate(el => document.activeElement === el), true);
    results.push({ check: 'SSE rendering, tool badge, markdown, ticket/refund dialog and restored focus', passed: true });
    await page.screenshot({ path: path.join(output, 'chat-response.png') });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('#history-toggle').click();
    assert.equal(await page.locator('#history-toggle').getAttribute('aria-expanded'), 'true');
    assert.equal(await page.locator('#sidebar').evaluate(el => el.inert), false);
    await page.screenshot({ path: path.join(output, 'mobile-history.png') });
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#sidebar').evaluate(el => el.inert), true);
    await page.locator('#new-chat').click();
    await page.locator('.empty h1').waitFor();
    await page.locator('#history-toggle').click();
    await page.locator('.conv-item').click();
    await page.locator('.bubble-text').filter({ hasText: '历史回答' }).waitFor();
    assert.equal(await page.locator('#history-toggle').getAttribute('aria-expanded'), 'false');
    results.push({ check: 'mobile history, Escape, and new conversation', passed: true });
    await page.route('**/api/review/queue*', route => route.fulfill({ json: { items: [{ id: 99, normalized_question: '审核弹窗界面检查', ai_suggested_answer: '用于验证界面，不写入知识库。', occurrence_count: 3, review_status: '待审' }] } }));
    await page.goto('http://localhost:8000/review', { waitUntil: 'networkidle' });
    const approve = page.locator('.ops .btn.ok');
    await approve.click();
    await page.locator('#mask.open').waitFor();
    assert.equal(await page.locator('.wrap').evaluate(el => el.inert), true);
    await page.locator('#modalAnswer').focus();
    await page.keyboard.press('Shift+Tab');
    assert.equal(await page.locator('#modalOk').evaluate(el => document.activeElement === el), true);
    await page.screenshot({ path: path.join(output, 'review-dialog-mobile.png') });
    await page.keyboard.press('Escape');
    assert.equal(await approve.evaluate(el => document.activeElement === el), true);
    results.push({ check: 'review dialog, focus trap, Escape and restored focus', passed: true });
    fs.writeFileSync(path.join(output, quick ? 'interaction-results.json' : 'results.json'), JSON.stringify({ errors, results }, null, 2));
    const overflows = results.filter(item => item.contentWidth > item.width);
    console.log(JSON.stringify({ layoutSnapshotsChecked: quick ? 0 : 24, syntax: 'passed', errors, overflows, interactions: results.filter(item => item.check), screenshots: output }, null, 2));
    assert.equal(errors.length, 0, 'Browser script errors');
    assert.equal(overflows.length, 0, 'Page horizontal overflow');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
