<script>
// ===== Simple DOM helper
const $ = (id) => document.getElementById(id);

// ===== I18N (same strings you already have, trimmed to essentials)
const I18N = {
  es: {
    appTitle: "Modbus Mirror • Panel en Vivo",
    gateTitle: "Ingrese PIN de Acceso",
    gateSub: "Por favor ingrese la contraseña para continuar.",
    gateEnter: "Entrar",
    gateErr: "PIN incorrecto. Inténtelo de nuevo.",
    userSettingsBtn: "Ajustes de Usuario",
    adminSettingsBtn: "Ajustes de Admin",
    ytLink: "Visite nuestro canal de YouTube",
    cfgTitle: "Configuración",
    cfgHint: "Todos los ajustes se recargan en caliente. TCP y RTU espejo reinician automáticamente.",
    reload: "Recargar",
    save: "Guardar",
    saved: "Guardado.",
    saveFail: "Fallo al guardar",
    reloaded: "Ajustes recargados",
    changePass: "Cambiar",
    enterOldGate: "Ingrese la contraseña actual del panel",
    enterNewGate: "Ingrese la nueva contraseña del panel",
    enterOldUser: "Ingrese la contraseña actual de Ajustes de Usuario",
    enterNewUser: "Ingrese la nueva contraseña de Ajustes de Usuario",
    wrongOld: "Contraseña actual incorrecta.",
    changedOk: "Contraseña cambiada correctamente.",
    tcpUnit1: "ID de Esclavo TCP (Modbus Poll, base 1)",
    modelBase: "RECTIFICADOR ADAX TECNA MODELO: {model} — N° SERIE: "
  },
  en: {
    appTitle: "Modbus Mirror • Live Dashboard",
    gateTitle: "Enter Access PIN",
    gateSub: "Please enter the password to continue.",
    gateEnter: "Enter",
    gateErr: "Incorrect PIN. Try again.",
    userSettingsBtn: "User Settings",
    adminSettingsBtn: "Admin Settings",
    ytLink: "Visit our YouTube channel",
    cfgTitle: "Configuration",
    cfgHint: "All settings hot-reload. TCP & Mirror RTU restart automatically.",
    reload: "Reload",
    save: "Save",
    saved: "Saved.",
    saveFail: "Save failed",
    reloaded: "Settings reloaded",
    changePass: "Change",
    enterOldGate: "Enter current dashboard password",
    enterNewGate: "Enter new dashboard password",
    enterOldUser: "Enter current User Settings password",
    enterNewUser: "Enter new User Settings password",
    wrongOld: "Wrong current password.",
    changedOk: "Password changed.",
    tcpUnit1: "TCP Slave Unit ID (Modbus Poll, 1-based)",
    modelBase: "RECTIFIER ADAX TECNA MODEL: {model} — SERIAL NO: "
  }
};

let LANG = localStorage.getItem("lang") || "es";
const t = (k, vars={}) => {
  const str = (I18N[LANG] && I18N[LANG][k]) || k;
  return str.replace(/\{(\w+)\}/g, (_,x)=> vars[x] ?? "");
};

// ===== PINs (local persistence)
const DEFAULT_GATE_PIN = "AT-MOD-01";
const DEFAULT_USER_PIN = "AT-User-1";
const ADMIN_SETTINGS_PIN = "AT1959";   // fixed

const LS_GATE_PIN = "dash_gate_pin";
const LS_USER_PIN = "dash_user_pin";
const SS_GATE_OK  = "dash_gate_ok_session"; // session unlock for dashboard

function getGatePin(){ return localStorage.getItem(LS_GATE_PIN) || DEFAULT_GATE_PIN; }
function setGatePin(v){ localStorage.setItem(LS_GATE_PIN, v); }

function getUserPin(){ return localStorage.getItem(LS_USER_PIN) || DEFAULT_USER_PIN; }
function setUserPin(v){ localStorage.setItem(LS_USER_PIN, v); }

// seed once
if (!localStorage.getItem(LS_GATE_PIN)) setGatePin(DEFAULT_GATE_PIN);
if (!localStorage.getItem(LS_USER_PIN)) setUserPin(DEFAULT_USER_PIN);

// migrate from old 1234/1111 one time
(function migrateOld(){
  if (localStorage.getItem(LS_GATE_PIN) === "1234") setGatePin(DEFAULT_GATE_PIN);
  if (localStorage.getItem(LS_USER_PIN) === "1111") setUserPin(DEFAULT_USER_PIN);
})();

// ===== Auth gates
function setGateSession(ok){ sessionStorage.setItem(SS_GATE_OK, ok ? "1":"0"); }
function isGateSessionOk(){ return sessionStorage.getItem(SS_GATE_OK) === "1"; }

// Require that dashboard session is unlocked, else go to /index.html
function requireDashboardUnlocked(){
  if (!isGateSessionOk()) location.replace("./index.html");
}

// Require role-specific PIN on settings pages
async function requireSettingsRole(role){
  // Simple prompt gate; you can swap to a modal if you prefer
  const entered = prompt(role === "admin" ? "Admin PIN" : "User Settings PIN");
  if (entered === null) { history.back(); return Promise.reject("cancel"); }
  if (role === "admin") {
    if (entered !== ADMIN_SETTINGS_PIN) { alert(I18N[LANG].gateErr); return requireSettingsRole(role); }
  } else {
    if (entered !== getUserPin()) { alert(I18N[LANG].gateErr); return requireSettingsRole(role); }
  }
  return true;
}

// ===== Utilities
function toast(msg){
  let el = $("toast");
  if (!el) { console.warn("toast element missing"); return; }
  el.textContent = msg;
  el.style.display = "block";
  setTimeout(()=> el.style.display="none", 2200);
}

// set language selector if present
function wireLanguage(selectId="langSel"){
  const sel = $(selectId);
  if (!sel) return;
  sel.value = LANG;
  sel.addEventListener("change", ()=> {
    LANG = sel.value || "es";
    localStorage.setItem("lang", LANG);
    location.reload(); // simplest refresh to re-apply text
  });
}
</script>
