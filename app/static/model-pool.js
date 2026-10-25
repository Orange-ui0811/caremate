(() => {
  const labels = {
    global_limit: ['总并发上限', 1, 512, 1], key_limit: ['单 Key 并发', 1, 128, 1],
    background_limit: ['后台并发上限', 0, 511, 1], max_queue: ['最大等待数', 1, 4096, 1],
    queue_timeout: ['最长排队（秒）', .05, 120, .05], call_timeout: ['单次调用期限（秒）', .1, 600, .1],
    read_timeout: ['连续无数据超时（秒）', .05, 300, .05], max_attempts: ['最多尝试次数', 1, 5, 1],
  };
  const $ = id => document.getElementById(id);
  const name = {chat: '聊天', embed: '向量化', rerank: '重排'};
  let loading = false, initialized = false;
  async function api(path, options) {
    const response = await fetch('/api/model-pool' + path, options);
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '设置无效，请检查后台并发小于总并发、读取超时不大于调用期限');
    return data;
  }
  function cell(row, text) { const node = document.createElement('td'); node.textContent = text; row.append(node); return node; }
  function rows(target, values, render) {
    $(target).replaceChildren();
    for (const value of values) { const tr = document.createElement('tr'); render(tr, value); $(target).append(tr); }
    if (!values.length) { const tr = document.createElement('tr'); cell(tr, '暂无记录'); $(target).append(tr); }
  }
  async function load() {
    if (loading) return;
    loading = true;
    try {
      const data = await api('/runtime');
      $('error').textContent = '';
      $('stats').replaceChildren();
      for (const [label, value] of [['运行中', data.running.length], ['等待中', data.waiting], ['已配置 Key', data.slots.length], ['可回放结果', data.replay_counts.SUCCEEDED || 0]]) {
        const node = document.createElement('div'), number = document.createElement('strong');
        number.textContent = value; node.append(number, document.createTextNode(label)); $('stats').append(node);
      }
      $('scope').textContent = `${data.enabled ? '调用池已启用' : '调用池已关闭'} · 总并发 ${data.policy.global_limit} · 单 Key ${data.policy.key_limit} · 整轮期限 ${data.turn_timeout} 秒 · 回放保留 ${data.replay_ttl / 3600} 小时`;
      if (!initialized) {
        for (const [key, [label, min, max, step]] of Object.entries(labels)) {
          const wrapper = document.createElement('label'), input = document.createElement('input');
          wrapper.textContent = label; Object.assign(input, {type: 'number', name: key, min, max, step, value: data.policy[key], required: true});
          wrapper.append(input); $('fields').append(wrapper);
        }
        initialized = true;
      }
      rows('slots', data.slots, (tr, slot) => {
        const state = slot.disabled ? '已停用' : slot.invalid ? '鉴权失效' : slot.cooldown > Date.now()/1000 ? '冷却中' : '可用';
        cell(tr, slot.resource); cell(tr, slot.channels.map(c => name[c]).join(' / ') + (slot.fallback ? '（备用）' : '')); cell(tr, state);
        cell(tr, data.running.filter(r => r.resource === slot.resource).length);
        const td = cell(tr, '');
        for (const [text, disabled] of [['恢复 / 重置', false], ['停用', true]]) {
          const btn = document.createElement('button'); btn.className = 'btn'; btn.textContent = text;
          btn.onclick = async () => { try { await api('/slots/' + slot.resource, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({disabled})}); await load(); } catch (e) { $('error').textContent = e.message; } };
          td.append(btn);
        }
        if (slot.managed) {
          const remove = document.createElement('button'); remove.className = 'btn'; remove.textContent = '移除';
          remove.onclick = async () => {
            if (!confirm(`移除 Key ${slot.resource}？新调用将不再使用它。`)) return;
            try { await api('/keys/' + slot.resource, {method: 'DELETE'}); await load(); }
            catch (e) { $('error').textContent = e.message; }
          };
          td.append(remove);
        }
      });
      rows('calls', data.recent_calls, (tr, call) => {
        cell(tr, call.request_id); cell(tr, (name[call.channel] || call.channel) + (call.priority === 'background' ? ' · 后台' : ''));
        cell(tr, call.model); cell(tr, call.resource); cell(tr, call.attempt);
        cell(tr, `${(call.queue_ms/1000).toFixed(2)}s / ${((call.finished || Date.now()/1000)-call.started).toFixed(2)}s`);
        cell(tr, call.outcome + (call.http_status ? ' · ' + call.http_status : ''));
      });
    } catch (error) { $('error').textContent = error.message; }
    finally { loading = false; }
  }
  $('policy').onsubmit = async event => {
    event.preventDefault(); const body = Object.fromEntries([...new FormData(event.target)].map(([k,v]) => [k, Number(v)]));
    try { await api('/policy', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}); await load(); $('error').textContent = '设置已保存，对新调用生效。'; }
    catch (e) { $('error').textContent = e.message; }
  };
  $('key-form').onsubmit = async event => {
    event.preventDefault();
    const keys = $('new-keys').value.split(/[\s,;]+/).filter(Boolean);
    const channels = [...document.querySelectorAll('[name="key-channel"]:checked')].map(input => input.value);
    if (!keys.length || !channels.length) { $('error').textContent = '请输入 Key 并选择用途。'; return; }
    const button = $('add-keys'); button.disabled = true;
    try {
      const result = await api('/keys', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({keys, channels})});
      $('new-keys').value = ''; await load(); $('error').textContent = `已保存 ${result.resources.length} 个 Key，对新调用立即生效。`;
    } catch (e) { $('error').textContent = e.message; }
    finally { button.disabled = false; }
  };
  mountAdminNav('/model-pool'); $('refresh').onclick = load; load();
  setInterval(() => { if (!document.hidden) load(); }, 5000);
})();
