<script>
async function fillSettings(role){
  try{
    const js = await apiGet("/api/settings");
    // topbar preview (optional)
    const b=js.branding||{}, d=js.device||{};
    if ($("phoneLink")) { $("phoneLink").textContent=b.phone||""; $("phoneLink").href=b.phone?("tel:"+b.phone.replace(/[^\d+]/g,'')):"#"; }
    if ($("emailLink")) { $("emailLink").textContent=b.email||""; $("emailLink").href=b.email?("mailto:"+b.email):"#"; }
    if ($("logoImg")) { const logo=b.logo_url||"/static/adax_logo.png"; $("logoImg").src=logo+(logo.includes("?")?"":"?v="+Date.now()); }
    if ($("ytLink")) { $("ytLink").textContent=t("ytLink"); if(b.youtube) $("ytLink").href=b.youtube; }

    // fields
    const u = js.upstream || {};
    const m = js.mirror_rtu || {};
    const tcfg = js.tcp || {};
    const lu = js.local_units || {};
    const dev = js.device || {};

    if ($("ups_uid"))  $("ups_uid").value  = (u.device_unit_id ?? 1);
    if ($("ups_poll")) $("ups_poll").value = (u.poll_period_s ?? 1.0);

    if ($("mir_slave")) $("mir_slave").value=(m.slave_id ?? (lu.unit1_id ?? 2));
    if ($("mir_baud"))  $("mir_baud").value=m.baudrate ?? 9600;
    if ($("mir_parity"))$("mir_parity").value=m.parity ?? "N";
    if ($("mir_stop"))  $("mir_stop").value=m.stopbits ?? 1;
    if ($("mir_byte"))  $("mir_byte").value=m.bytesize ?? 8;

    if ($("tcp_port"))  $("tcp_port").value=tcfg.port ?? 1502;
    if ($("tcp_unit1")) $("tcp_unit1").value=(lu.unit1_id ?? 2);

    // branding & device (admin only UI)
    if (role==="admin"){
      if ($("brand_phone")) $("brand_phone").value=b.phone||"";
      if ($("brand_email")) $("brand_email").value=b.email||"";
      if ($("brand_youtube")) $("brand_youtube").value=b.youtube||"";
      if ($("brand_logo")) $("brand_logo").value=b.logo_url||"";
      if ($("brand_qr")) $("brand_qr").value=b.qr_url||"";
      if ($("device_model")) $("device_model").value=dev.model||"";
    }
  }catch(e){ console.error(e); }
}

function safeNum(id, fallback){
  const el = $(id);
  if (!el) return fallback;
  const n = +el.value;
  return Number.isFinite(n) ? n : fallback;
}

async function saveSettings(role){
  const basePayload = {
    upstream: {
      device_unit_id: safeNum("ups_uid", 1),
      poll_period_s:  safeNum("ups_poll", 1.0)
    },
    mirror_rtu: {
      slave_id: safeNum("mir_slave", 2),
      baudrate: safeNum("mir_baud", 9600),
      parity: $("mir_parity")?.value || "N",
      stopbits: safeNum("mir_stop", 1),
      bytesize: safeNum("mir_byte", 8)
    },
    tcp: { port: safeNum("tcp_port", 1502) },
    local_units: { unit1_id: safeNum("tcp_unit1", 2) },
    hr: { start: 0, count: 24 }
  };

  let payload = basePayload;
  if (role==="admin"){
    payload = {
      ...basePayload,
      branding:{
        phone:$("brand_phone")?.value||"",
        email:$("brand_email")?.value||"",
        youtube:$("brand_youtube")?.value||"",
        logo_url:$("brand_logo")?.value||"",
        qr_url:$("brand_qr")?.value||""
      },
      device:{ model:$("device_model")?.value||"" }
    };
  }

  const btn=$("btnSave"); if (btn){ btn.disabled=true; const old=btn.textContent; btn.textContent=t("save");
    try{
      const js = await apiPut("/api/settings", payload);
      if (js.ok){ toast(t("saved")); await fillSettings(role); }
      else { toast(js.detail||t("saveFail")); }
    }catch(_){ toast(t("saveFail")); }
    finally{ btn.disabled=false; btn.textContent=old; }
  } else {
    try{
      const js = await apiPut("/api/settings", payload);
      if (js.ok){ alert("Saved"); } else { alert("Save failed"); }
    }catch(_){ alert("Save failed"); }
  }
}

function wireSettingsPage(role){
  wireLanguage();
  $("btnBack")?.addEventListener("click", ()=> location.href="./dashboard.html");
  $("btnReload")?.addEventListener("click", ()=> fillSettings(role));
  $("btnSave")?.addEventListener("click", ()=> saveSettings(role));

  // user can change their own settings PIN via a small prompt flow
  $("pinChangeBtn")?.addEventListener("click", ()=>{
    if (role!=="user") return;
    const oldP = prompt(t("enterOldUser"));
    if (oldP===null) return;
    if (oldP !== getUserPin()){ alert(t("wrongOld")); return; }
    const newP = prompt(t("enterNewUser"));
    if (!newP || !newP.trim()) return;
    setUserPin(newP.trim());
    alert(t("changedOk"));
  });
}
</script>
