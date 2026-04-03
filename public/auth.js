/**
 * auth.js — Login modal for Microsoft Entra ID
 *
 * Intercepts clicks on the "#login" header link and shows a styled
 * modal with a "Iniciar Sesión con Microsoft" button. When clicked,
 * it calls /auth/login. If Entra is not configured, an error is shown.
 */

(function () {
  "use strict";

  /* ───────── helpers ───────── */

  function createModal() {
    // Backdrop
    const backdrop = document.createElement("div");
    backdrop.id = "login-modal-backdrop";
    backdrop.className = "login-modal-backdrop";

    // Modal card
    const modal = document.createElement("div");
    modal.className = "login-modal";

    // Close button
    const closeBtn = document.createElement("button");
    closeBtn.className = "login-modal-close";
    closeBtn.innerHTML = "&times;";
    closeBtn.title = "Cerrar";
    closeBtn.addEventListener("click", () => destroyModal());

    // Title
    const title = document.createElement("h2");
    title.className = "login-modal-title";
    title.textContent = "Iniciar Sesión";

    // Subtitle
    const subtitle = document.createElement("p");
    subtitle.className = "login-modal-subtitle";
    subtitle.textContent =
      "Vincula tu cuenta institucional UPY para una experiencia personalizada.";

    // Divider
    const divider = document.createElement("div");
    divider.className = "login-modal-divider";
    const dividerLine1 = document.createElement("span");
    const dividerText = document.createElement("span");
    dividerText.textContent = "continuar con";
    const dividerLine2 = document.createElement("span");
    divider.append(dividerLine1, dividerText, dividerLine2);

    // Microsoft button
    const msBtn = document.createElement("button");
    msBtn.className = "login-ms-btn";
    msBtn.id = "login-ms-btn";
    msBtn.innerHTML =
      '<img src="/public/microsoft_logo.svg" alt="Microsoft" width="20" height="20" />' +
      "<span>Iniciar Sesión con Microsoft</span>";
    msBtn.addEventListener("click", handleMicrosoftLogin);

    // Error container (hidden by default)
    const errorBox = document.createElement("div");
    errorBox.className = "login-modal-error";
    errorBox.id = "login-modal-error";
    errorBox.style.display = "none";

    // Footer
    const footer = document.createElement("p");
    footer.className = "login-modal-footer";
    footer.textContent = "El inicio de sesión es opcional. Puedes seguir usando el chat sin autenticarte.";

    modal.append(closeBtn, title, subtitle, divider, msBtn, errorBox, footer);
    backdrop.appendChild(modal);

    // Close on backdrop click
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) destroyModal();
    });

    // Close on Escape
    document.addEventListener("keydown", handleEscape);

    document.body.appendChild(backdrop);

    // Animate in
    requestAnimationFrame(() => {
      backdrop.classList.add("login-modal-visible");
    });
  }

  function destroyModal() {
    const backdrop = document.getElementById("login-modal-backdrop");
    if (!backdrop) return;
    backdrop.classList.remove("login-modal-visible");
    document.removeEventListener("keydown", handleEscape);
    setTimeout(() => backdrop.remove(), 250);
  }

  function handleEscape(e) {
    if (e.key === "Escape") destroyModal();
  }

  function showError(message) {
    const errorBox = document.getElementById("login-modal-error");
    if (!errorBox) return;
    errorBox.textContent = message;
    errorBox.style.display = "block";
  }

  async function handleMicrosoftLogin() {
    const btn = document.getElementById("login-ms-btn");
    if (btn) {
      btn.disabled = true;
      btn.querySelector("span").textContent = "Conectando…";
    }

    try {
      const response = await fetch("/auth/login", {
        method: "GET",
        headers: { Accept: "application/json" },
      });

      const data = await response.json();

      if (!response.ok || data.error) {
        showError(
          data.detail ||
            data.error ||
            "Error: No se pudo conectar con Microsoft Entra ID."
        );
        if (btn) {
          btn.disabled = false;
          btn.querySelector("span").textContent =
            "Iniciar Sesión con Microsoft";
        }
        return;
      }

      // If we got an auth URL, open it in a new tab
      if (data.auth_url) {
        window.open(data.auth_url, "_blank");
        destroyModal();
      }
    } catch (err) {
      showError("Error de red. Verifica tu conexión e intenta de nuevo.");
      if (btn) {
        btn.disabled = false;
        btn.querySelector("span").textContent = "Iniciar Sesión con Microsoft";
      }
    }
  }

  /* ───────── intercept header link ───────── */

  function interceptLoginLink() {
    document.addEventListener("click", (e) => {
      const link = e.target.closest('a[href="#login"]');
      if (link) {
        e.preventDefault();
        e.stopPropagation();
        createModal();
      }
    });
  }

  /* ───────── init ───────── */

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", interceptLoginLink);
  } else {
    interceptLoginLink();
  }
})();
