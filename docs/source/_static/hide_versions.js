document.addEventListener("DOMContentLoaded", function () {
  document
    .querySelectorAll(".rst-versions, .rst-current-version, .switch-menus, .floating.container.bottom-right")
    .forEach(el => el.remove());
});
