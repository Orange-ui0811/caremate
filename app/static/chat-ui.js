/* Presentation-only enhancements: mobile history and accessible modal focus. */
(() => {
  const app = document.querySelector('.app');
  const sidebar = document.getElementById('sidebar');
  const toggle = document.getElementById('history-toggle');
  const backdrop = document.querySelector('.history-backdrop');
  const mobile = matchMedia('(max-width: 680px)');
  const focusable = root => [...root.querySelectorAll('button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled)')].filter(el => el.getClientRects().length && !el.closest('[inert]'));
  function setHistory(open) {
    app.classList.toggle('history-open', open);
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? '关闭会话记录' : '打开会话记录');
    sidebar.inert = mobile.matches && !open;
    if (open) (focusable(sidebar)[0] || toggle).focus();
    else if (mobile.matches) toggle.focus();
  }
  toggle.addEventListener('click', () => setHistory(!app.classList.contains('history-open')));
  backdrop.addEventListener('click', () => setHistory(false));
  sidebar.addEventListener('click', event => { if (mobile.matches && event.target.closest('.conv-item')) setHistory(false); });
  mobile.addEventListener('change', () => { app.classList.remove('history-open'); sidebar.inert = mobile.matches; toggle.setAttribute('aria-expanded', 'false'); });
  sidebar.inert = mobile.matches;

  const main = document.getElementById('chat-main');
  const masks = [...document.querySelectorAll('.modal-mask')];
  const previousFocus = new WeakMap();
  let lastOutsideFocus = document.activeElement;
  document.addEventListener('focusin', event => {
    if (!event.target.closest('.modal-mask')) lastOutsideFocus = event.target;
  });
  for (const mask of masks) {
    mask.setAttribute('aria-hidden', 'true');
    const observer = new MutationObserver(() => {
      const open = mask.classList.contains('open');
      mask.setAttribute('aria-hidden', String(!open));
      if (open) {
        previousFocus.set(mask, lastOutsideFocus);
        main.inert = true;
        sidebar.inert = true;
        (focusable(mask)[0] || mask.querySelector('[role=dialog]')).focus();
      } else {
        main.inert = masks.some(item => item.classList.contains('open'));
        sidebar.inert = main.inert || (mobile.matches && !app.classList.contains('history-open'));
        const previous = previousFocus.get(mask);
        if (previous?.isConnected) previous.focus();
        else if (mobile.matches && app.classList.contains('history-open')) (focusable(sidebar)[0] || toggle).focus();
        else document.getElementById('input').focus();
      }
    });
    observer.observe(mask, { attributes: true, attributeFilter: ['class'] });
  }
  document.addEventListener('keydown', event => {
    const mask = masks.find(item => item.classList.contains('open'));
    if (event.key === 'Escape') {
      if (mask) { mask.querySelector('.cancel:not(:disabled)')?.click(); }
      else if (app.classList.contains('history-open')) setHistory(false);
    }
    if (event.key !== 'Tab') return;
    const target = mask || (mobile.matches && app.classList.contains('history-open') ? sidebar : null);
    if (!target) return;
    const elements = focusable(target);
    const first = elements[0], last = elements.at(-1);
    if (!first) { if (!mask) setHistory(false); return; }
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  });
})();
