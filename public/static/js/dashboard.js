<script>
// Minimal version using your previous logic

const names = [
  "BATTERY_VOLTAGE_V","LOAD_VOLTAGE_V","BATTERY_CURRENT_A","LOAD_CURRENT_A","TOTAL_CURRENT_A",
  "AC_VOLTAGE_RN_raw","AC_VOLTAGE_SN_raw","AC_VOLTAGE_TN_raw",
  "AMBIENT_TEMP_C","AMBIENT_TEMP_MAX_C","ALARM_BYTE_1_bits","ALARM_BYTE_2_bits","BATTERY_MODE_raw",
  "BATTERY_TEST_COUNT","LAST_TEST_DURATION_H","LAST_TEST_DURATION_MIN","LAST_TEST_DAY","LAST_TEST_MONTH","LAST_TEST_YEAR",
  "LAST_TEST_FINAL_BATT_V","FLOAT_CURRENT_SETPOINT_PbCa_A","NUM_CELLS_NiCd","TIMER_HOURS_NiCd","SERIAL_NUMBER_raw"
];

const F=(n,u)=>typeof n==="number"&&isFinite(n)?n.toFixed(1)+" "+(u||""):"--";
function pill2(text, color){ return `<span class="pill2 ${color}">${text}</span>`; }

function applyModelLine(model, serialRaw){
  const base = t("modelBase", {model: model || "CEA"});
  $("modelLine").textContent = base + (serialRaw!=="" ? serialRaw : "—");
  $("modelLine").dataset.base = base;
}

function renderTopbarFromSettings(js){
  const b=js.branding||{}, d=js.device||{};
  const phone=b.phone||"", email=b.email||"", yt=b.youtube||"", logo=b.logo_url||"/static/adax_logo.png", qrUrl=b.qr_url||"", model=d.model||"CEA";
  $("phoneLink").textContent=phone; $("phoneLink").href=phone?("tel:"+phone.replace(/[^\d+]/g,'')):"#";
  $("emailLink").textContent=email; $("emailLink").href=email?("mailto:"+email):"#";
  $("ytLink").textContent = t("ytLink");
  if(logo){ $("logoImg").src=logo+(logo.includes("?")?"":"?v="+Date.now()); $("logoImg").style.display="block"; } else {$("logoImg").style.display="none";}
  if(yt) $("ytLink").href=yt; else $("ytLink").removeAttribute("href");
  const finalQR = qrUrl || (yt?("https://api.qrserver.com/v1/create-qr-code/?size=120x120&data="+encodeURIComponent(yt)):"");
  if(finalQR){ $("qrImg").src=finalQR; $("qrImg").style.display="block"; } else { $("qrImg").style.display="none"; }
  applyModelLine(model, "");
}

async function tickTiles(){
  try{
    const js = await apiGet("/api/hr");
    const s=js.scaled||{};
    $("vb").textContent=F(s.BATTERY_VOLTAGE_V,"Vdc");
    $("vl").textContent=F(s.LOAD_VOLTAGE_V,"Vdc");
    $("ib").textContent=F(s.BATTERY_CURRENT_A,"A");
    $("il").textContent=F(s.LOAD_CURRENT_A,"A");
    $("it").textContent=F(s.TOTAL_CURRENT_A,"A");
    $("ta").textContent=F(s.AMBIENT_TEMP_C,"°C");
    $("m").textContent="modo: "+(js.mode||"--");
    $("t").textContent="último: "+new Date().toLocaleTimeString();

    const raw=js.raw||[];
    const serialRaw = (raw.length>23 ? raw[23] : "");
    const base = $("modelLine").dataset.base || t("modelBase",{model:"CEA"});
    $("modelLine").textContent = base + (serialRaw!=="" ? serialRaw : "—");

    const tb=document.querySelector("#tbl tbody"); if (!tb) return;
    tb.innerHTML="";
    names.forEach((nm,i)=>{
      const tr=document.createElement("tr");
      const td=(x)=>{const d=document.createElement("td"); d.textContent=x; return d;}
      tr.appendChild(td(i));
      tr.appendChild(td(nm));
      tr.appendChild(td(raw[i]??""));
      tr.appendChild(td(typeof s[nm]==="number"?s[nm]:""));
      tb.appendChild(tr);
    });
  }catch(e){}
}

async function refreshTopCards(){
  try{
    const [measR, alarmsR, statesR] = await Promise.allSettled([ apiGet("/api/meas2"), apiGet("/api/alarms2"), apiGet("/api/states2") ]);
    if (measR.status==="fulfilled"){
      const m = measR.value;
      const rows=[
        ["BATTERY VOLTAGE", `${(m.battery_voltage_v??0).toFixed(1)} Vdc`],
        ["LOAD VOLTAGE", `${(m.load_voltage_v??0).toFixed(1)} Vdc`],
        ["BATTERY CURRENT", `${(m.battery_current_a??0).toFixed(1)} A`],
        ["LOAD CURRENT", `${(m.load_current_a??0).toFixed(1)} A`],
        ["TOTAL CURRENT", `${(m.total_current_a??0).toFixed(1)} A`],
        ['AC "R-N"', `${m.ac_rn_v??0} Vac`],
        ['AC "S-N"', `${m.ac_sn_v??0} Vac`],
        ['AC "T-N"', `${m.ac_tn_v??0} Vac`],
        ["AMBIENT TEMPERATURE", `${(m.ambient_temp_c??0).toFixed(1)} °C`],
        ["MAX RECORDED TEMP", `${(m.ambient_temp_max_c??0).toFixed(1)} °C`],
      ];
      $("measGrid").innerHTML=rows.map(([k,v])=>`<div>${k}</div><div class="mono" style="font-weight:600">${v}</div>`).join("");
    }
    if (alarmsR.status==="fulfilled"){
      const a = alarmsR.value.items||[];
      $("alarmsList").innerHTML = a.map(it=>{
        const status = it.active ? pill2("ACTIVE","red") : pill2("NORMAL","green");
        return `<div class="kv"><div>${it.label||it.key}</div><div>${status}</div></div>`;
      }).join("");
    }
    if (statesR.status==="fulfilled"){
      const s = statesR.value.items||[];
      $("statesList").innerHTML = s.map(it=>`<div class="kv"><div>${it.label||it.key}</div><div>${pill2(it.value, "black")}</div></div>`).join("");
    }
  }catch(e){}
}

async function bootDashboard(){
  requireDashboardUnlocked();
  wireLanguage();
  // topbar
  try { const js = await apiGet("/api/settings"); renderTopbarFromSettings(js); } catch(_){}
  // buttons → navigate to dedicated pages
  const bUser = $("btnUserSettings"), bAdmin = $("btnAdminSettings");
  if (bUser) bUser.onclick = ()=> location.href = "./settings-user.html";
  if (bAdmin) bAdmin.onclick = ()=> location.href = "./settings-admin.html";

  // loops
  const loop = async ()=>{ await Promise.all([tickTiles(), refreshTopCards()]); setTimeout(loop, 1000); };
  loop();
}

document.addEventListener("DOMContentLoaded", bootDashboard);
</script>
