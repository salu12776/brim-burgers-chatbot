/*
 * Brim Burgers chat widget: a floating button that opens the chatbot in a panel.
 * Add ONE line before </body> on the client's website:
 *
 *   <script src="https://YOUR-SERVER/static/widget.js" defer></script>
 *
 * The chat runs from the same server as this file, so add the website's origin
 * to ALLOWED_ORIGINS in .env only if you call /chat directly from that site.
 */
(function () {
  if (window.__brimWidget) return;
  window.__brimWidget = true;

  var script = document.currentScript;
  var base = script ? new URL(script.src).origin : "";
  var RED = "#C8102E", MUSTARD = "#F5B82E";

  var css = document.createElement("style");
  css.textContent =
    ".brim-launch{position:fixed;right:20px;bottom:20px;z-index:2147483000;width:62px;height:62px;border-radius:50%;" +
    "border:0;background:" + RED + ";color:#fff;cursor:pointer;box-shadow:0 8px 24px rgba(120,10,30,.35);display:grid;place-items:center;" +
    "transition:transform .15s ease}" +
    ".brim-launch:hover{transform:scale(1.06)}.brim-launch:focus-visible{outline:3px solid " + MUSTARD + ";outline-offset:3px}" +
    ".brim-launch svg{width:28px;height:28px}" +
    ".brim-frame{position:fixed;right:20px;bottom:96px;z-index:2147483000;width:400px;height:min(640px,calc(100vh - 120px));" +
    "border:0;border-radius:20px;box-shadow:0 18px 50px rgba(43,29,22,.28);background:#FFF4DC;opacity:0;transform:translateY(12px);" +
    "pointer-events:none;transition:opacity .18s ease,transform .18s ease}" +
    ".brim-frame.open{opacity:1;transform:none;pointer-events:auto}" +
    "@media (max-width:520px){.brim-frame{right:0;bottom:0;width:100vw;height:100dvh;border-radius:0}" +
    ".brim-frame.open~.brim-launch{bottom:auto;top:14px;right:14px;width:44px;height:44px;box-shadow:none;background:transparent}}" +
    "@media (prefers-reduced-motion:reduce){.brim-frame,.brim-launch{transition:none}}";
  document.head.appendChild(css);

  var frame = document.createElement("iframe");
  frame.className = "brim-frame";
  frame.title = "Brim Burgers assistant";
  frame.setAttribute("aria-hidden", "true");
  frame.setAttribute("tabindex", "-1");

  var CHAT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/></svg>';
  var CLOSE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>';

  var btn = document.createElement("button");
  btn.className = "brim-launch";
  btn.type = "button";
  btn.setAttribute("aria-label", "Chat with Brim Burgers");
  btn.setAttribute("aria-expanded", "false");
  btn.innerHTML = CHAT;

  var open = false;
  btn.addEventListener("click", function () {
    open = !open;
    if (open && !frame.src) frame.src = base + "/?embed=1";
    frame.classList.toggle("open", open);
    frame.setAttribute("aria-hidden", String(!open));
    frame.setAttribute("tabindex", open ? "0" : "-1");
    btn.setAttribute("aria-expanded", String(open));
    btn.setAttribute("aria-label", open ? "Close chat" : "Chat with Brim Burgers");
    btn.innerHTML = open ? CLOSE : CHAT;
    if (open) setTimeout(function () { try { frame.focus(); } catch (e) {} }, 200);
  });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape" && open) btn.click(); });

  document.body.appendChild(frame);
  document.body.appendChild(btn);
})();