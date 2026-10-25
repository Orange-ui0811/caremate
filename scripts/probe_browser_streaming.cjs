// Measure real transport chunks and visible text growth without changing app rendering.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const uid = 'sse-probe-' + crypto.randomUUID();
  let cid;
  try {
    const page = await browser.newPage();
    await page.addInitScript(user => {
      localStorage.setItem('mewhelp_session_id', user);
      localStorage.removeItem('mewhelp_conversation_id');
      const p = window.streamProbe = { chunks: [], deltas: [], paints: [], started: null, finished: false };
      const originalFetch = window.fetch.bind(window);
      window.fetch = async (...args) => {
        const isChat = String(args[0]).endsWith('/api/chat');
        if (isChat) p.started = performance.now();
        const response = await originalFetch(...args);
        if (!isChat) return response;
        p.headersAt = performance.now() - p.started;
        p.contentType = response.headers.get('content-type');
        const getReader = response.body.getReader.bind(response.body);
        response.body.getReader = (...opts) => {
          const reader = getReader(...opts), read = reader.read.bind(reader), decoder = new TextDecoder();
          let buffer = '';
          reader.read = async (...readArgs) => {
            const result = await read(...readArgs);
            const ms = performance.now() - p.started;
            if (result.value) {
              p.chunks.push({ ms, bytes: result.value.length });
              buffer += decoder.decode(result.value, { stream: true });
              const frames = buffer.split('\n\n'); buffer = frames.pop();
              for (const frame of frames) {
                const line = frame.split('\n').find(l => l.startsWith('data: '));
                if (!line || line === 'data: [DONE]') continue;
                const data = JSON.parse(line.slice(6));
                if (data.delta) p.deltas.push({ ms, chars: data.delta.length });
                if (data.event === 'done') { p.conversationId = data.conversation_id; p.doneAt = ms; }
              }
            }
            return result;
          };
          return reader;
        };
        return response;
      };
      let last = 0;
      function sample() {
        const chars = document.querySelector('.row.bot .bubble-text')?.textContent.length || 0;
        if (p.started != null && chars !== last) {
          p.paints.push({ ms: performance.now() - p.started, chars }); last = chars;
        }
        if (!p.finished) requestAnimationFrame(sample);
      }
      requestAnimationFrame(sample);
    }, uid);
    await page.goto('http://127.0.0.1:8000/');
    await page.locator('#input').fill('退货换货政策是怎么样的');
    await page.locator('#send').click();
    await page.waitForFunction(() => window.streamProbe.started !== null && !document.getElementById('send').disabled, null, { timeout: 180000 });
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => { window.streamProbe.finished = true; resolve(); })));
    const result = await page.evaluate(() => window.streamProbe);
    cid = result.conversationId;
    result.summary = { chunks: result.chunks.length, deltaCount: result.deltas.length,
      firstTextSeconds: result.deltas[0]?.ms / 1000, lastTextSeconds: result.deltas.at(-1)?.ms / 1000,
      textDeliverySeconds: ((result.deltas.at(-1)?.ms || 0) - (result.deltas[0]?.ms || 0)) / 1000,
      visibleGrowthFrames: result.paints.length, finalTextChars: result.paints.at(-1)?.chars };
    const out = path.join(root, 'data/verification/sse-browser-20261016.json');
    fs.writeFileSync(out, JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ output: out, contentType: result.contentType, conversationId: cid, summary: result.summary }));
    if (!cid) throw new Error('No completed answer; cannot conclude streaming status');
  } finally {
    if (cid) {
      const response = await fetch(`http://127.0.0.1:8000/api/conversations/${cid}?user_id=${uid}`, { method: 'DELETE' });
      if (!response.ok) console.error('Test conversation cleanup failed:', response.status);
    }
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
