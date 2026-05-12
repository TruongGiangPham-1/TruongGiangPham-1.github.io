// Theme toggle. The no-flash init script in head.html has already
// set data-theme on <html>; this just handles clicks afterwards.
(function () {
  var button = document.querySelector('.theme-toggle');
  if (!button) return;

  button.addEventListener('click', function () {
    var current = document.documentElement.getAttribute('data-theme');
    var next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('theme', next); } catch (e) {}
  });
})();
