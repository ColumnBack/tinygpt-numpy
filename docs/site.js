(function () {
  var root = document.documentElement;
  function setLang(l) {
    root.setAttribute("data-lang", l);
    root.lang = l;
    document.querySelectorAll("[data-set-lang]").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-set-lang") === l));
    });
    try { localStorage.setItem("tinygpt-lang", l); } catch (e) {}
  }
  var initial = "en";
  var hash = (location.hash || "").replace("#", "");
  var saved = null;
  try { saved = localStorage.getItem("tinygpt-lang"); } catch (e) {}
  if (hash === "ko" || hash === "en") initial = hash;
  else if (saved === "ko" || saved === "en") initial = saved;
  else if ((navigator.language || "").toLowerCase().indexOf("ko") === 0) initial = "ko";
  setLang(initial);
  document.querySelectorAll("[data-set-lang]").forEach(function (b) {
    b.addEventListener("click", function () { setLang(b.getAttribute("data-set-lang")); });
  });

  document.querySelectorAll(".copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = btn.getAttribute("data-copy");
      function done(msg) { btn.textContent = msg; setTimeout(function () { btn.textContent = "Copy"; }, 1500); }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { done("Copied"); }, function () { selectPre(btn); done("Selected"); });
      } else { selectPre(btn); done("Selected"); }
    });
  });
  function selectPre(btn) {
    var pre = btn.parentNode.querySelector("pre");
    var r = document.createRange(); r.selectNodeContents(pre);
    var s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
  }
})();
