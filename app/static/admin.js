/* 后台管理外壳:一份导航挂在所有后台页上(知识库录入 / RAG 评估 / 待审队列 / 观测与成本 /
   主题分布 / 分类器验收)。

   页面共用 apple-theme.css,导航在这里生成;
   整个文件包在 IIFE 里,只往 window 上挂两个名字,免得和宿主页的 $ / el 撞名。
   模块入口都是各章原本的路径,导航只是把它们收到一处,不做跳转改写——文档里贴的链接照样能用。 */
(function () {
  const NAV = [
    { href: "/admin", label: "工作台" },
    { href: "/kb", label: "知识库" },
    { href: "/rag-eval", label: "检索评估" },
    { href: "/review", label: "待审队列" },
    { href: "/observability", label: "观测与成本" },
    { href: "/model-pool", label: "模型调用池" },
    { href: "/topics", label: "主题分布" },
    {
      href: "/acceptance", label: "分类器",
      children: [
        ["/acceptance", "总览"],
        ["/acceptance/eval", "评测详情"],
        ["/acceptance/data", "数据产物"],
        ["/acceptance/errors", "错例复核"],
      ],
    },
  ];


  function link(href, label, on, cls) {
    const a = document.createElement("a");
    a.href = href;
    a.className = (on ? "on " : "") + (cls || "");
    a.textContent = label;
    if (on) a.setAttribute("aria-current", "page");
    return a;
  }

  /** 当前页归属哪个模块:精确命中优先,其次按前缀(/acceptance/eval 归 /acceptance)。 */
  function moduleOf(active) {
    return NAV.find((m) => m.href === active)
      || NAV.find((m) => m.href !== "/" && active.startsWith(m.href + "/"));
  }

  function renderAdminNav(active) {
    const wrap = document.createElement("nav");
    wrap.className = "mh-nav";
    wrap.setAttribute("aria-label", "管理端导航");
    const row = document.createElement("div");
    row.className = "row";
    const brand = document.createElement("a");
    brand.href = "/admin";
    brand.className = "brand-link";
    brand.innerHTML = '<img src="/static/brand.svg" width="23" height="23" alt="" />客小助 · CareMate';
    row.appendChild(brand);
    const mod = moduleOf(active);
    for (const m of NAV) row.appendChild(link(m.href, m.label, m === mod));
    const grow = document.createElement("span");
    grow.className = "grow";
    row.appendChild(grow);
    row.appendChild(link("/", "智能客服 ↗", false));
    wrap.appendChild(row);

    if (mod && mod.children) {
      const sub = document.createElement("div");
      sub.className = "sub";
      for (const [href, label] of mod.children) sub.appendChild(link(href, label, href === active));
      wrap.appendChild(sub);
    }
    return wrap;
  }

  /** 放到页面顶部,保持各管理模块导航位置一致。 */
  function mountAdminNav(active) {
    const nav = renderAdminNav(active);
    const bar = document.querySelector(".topbar");
    if (bar && bar.parentNode) bar.parentNode.insertBefore(nav, bar.parentNode.firstChild);
    else {
      const w = document.querySelector(".wrap") || document.body;
      w.insertBefore(nav, w.firstChild);
    }
    const updateScroll = () => nav.classList.toggle("scrolled", window.scrollY > 8);
    window.addEventListener("scroll", updateScroll, { passive: true });
    updateScroll();
    return nav;
  }

  window.renderAdminNav = renderAdminNav;
  window.mountAdminNav = mountAdminNav;

  // Keep the review dialog navigable with a keyboard and return to its trigger.
  const mask = document.querySelector('.mask');
  if (mask) {
    const workspace = document.querySelector('.wrap');
    let trigger = null, lastOutsideFocus = document.activeElement;
    mask.setAttribute('aria-hidden', 'true');
    document.addEventListener('focusin', event => {
      if (!mask.contains(event.target)) lastOutsideFocus = event.target;
    });
    new MutationObserver(() => {
      const open = mask.classList.contains('open');
      mask.setAttribute('aria-hidden', String(!open));
      workspace.inert = open;
      if (open) { trigger = lastOutsideFocus; mask.querySelector('textarea')?.focus(); }
      else trigger?.focus();
    }).observe(mask, { attributes: true, attributeFilter: ['class'] });
    document.addEventListener('keydown', event => {
      if (!mask.classList.contains('open')) return;
      if (event.key === 'Escape') document.getElementById('modalCancel')?.click();
      if (event.key !== 'Tab') return;
      const elements = [...mask.querySelectorAll('textarea, button:not(:disabled)')];
      const first = elements[0], last = elements.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    });
  }
})();
