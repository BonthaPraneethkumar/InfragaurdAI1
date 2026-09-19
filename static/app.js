const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
const state={stream:null,lat:null,lon:null,locationLabel:"",evidenceToken:null,redactions:{},audioBlob:null,recorder:null,voiceTimer:null,voiceSeconds:20,transcript:"",analysis:null};

function toast(msg){const t=$("#toast");t.textContent=msg;t.classList.add("show");setTimeout(()=>t.classList.remove("show"),2200)}
function escapeHtml(v=""){return String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]))}
async function api(url,opt={}){const r=await fetch(url,opt);const b=await r.json().catch(()=>({}));if(!r.ok)throw new Error(b.detail||b.message||`Request failed (${r.status})`);return b}

function go(name){
  $$(".screen").forEach(x=>x.classList.remove("active"));
  $(`#${name}Screen`)?.classList.add("active");
  const showCitizen=!["capture","officer","analytics"].includes(name);
  $("#bottomNav").classList.toggle("hidden",!showCitizen);
  $$("#bottomNav [data-go]").forEach(b=>b.classList.toggle("active",b.dataset.go===name));
  window.scrollTo({top:0,behavior:"smooth"});
  if(name==="myreports")loadReports();
  if(name==="notifications")loadNotifications();
  if(name==="officer")loadOfficer();
  if(name==="analytics")loadAnalytics();
}
$$("[data-go]").forEach(b=>b.addEventListener("click",()=>go(b.dataset.go)));

$("#reportIssueBtn").addEventListener("click",()=>{go("capture");preparePermissions()});
$("#navReportBtn").addEventListener("click",()=>{go("capture");preparePermissions()});

async function preparePermissions(){
  $("#permissionCard").classList.remove("hidden");
  $("#cameraStage").classList.add("hidden");
  $("#processingPanel").classList.add("hidden");
  $("#voiceStage").classList.add("hidden");
  $("#analysisStage").classList.add("hidden");
}

$("#startPermissionsBtn").addEventListener("click", async ()=>{
  try{
    // CAMERA permission is requested here, on the citizen's camera action.
    // MICROPHONE permission is intentionally requested later, when the voice step begins.
    // The browser/OS controls first-use permission prompts and they cannot be bypassed.
    const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:"environment"}},audio:false});
    state.stream=stream;
    $("#cameraVideo").srcObject=stream;
    $("#permissionCard").classList.add("hidden");
    $("#cameraStage").classList.remove("hidden");
    captureLocation().catch(()=>toast("Allow location once to attach GPS automatically."));
  }catch(e){toast(`Camera permission error: ${e.message}`)}
});

async function captureLocation(){
  if(!navigator.geolocation)throw new Error("Location is not available in this browser.");
  return new Promise((resolve,reject)=>{
    navigator.geolocation.getCurrentPosition(async p=>{
      state.lat=p.coords.latitude;state.lon=p.coords.longitude;
      $("#locationChip").textContent=`📍 ${state.lat.toFixed(5)}, ${state.lon.toFixed(5)}`;
      try{
        const r=await api(`/api/reverse-geocode?lat=${encodeURIComponent(state.lat)}&lon=${encodeURIComponent(state.lon)}`);
        state.locationLabel=r.label;
        $("#locationChip").textContent=`📍 ${r.label}`;
      }catch{state.locationLabel=`${state.lat.toFixed(6)}, ${state.lon.toFixed(6)}`}
      resolve();
    },reject,{enableHighAccuracy:true,timeout:15000,maximumAge:30000});
  });
}

$("#uploadFallbackBtn").addEventListener("click",()=>$("#fallbackFile").click());
$("#fallbackFile").addEventListener("change",async()=>{
  const file=$("#fallbackFile").files?.[0];if(!file)return;
  try{await captureLocation()}catch(e){toast("Location permission was not granted. You can continue, but GPS will be missing.")}
  await processPhoto(file);
});

$("#captureBtn").addEventListener("click",async()=>{
  const v=$("#cameraVideo");
  const canvas=document.createElement("canvas");
  canvas.width=v.videoWidth;canvas.height=v.videoHeight;
  canvas.getContext("2d").drawImage(v,0,0);
  const blob=await new Promise(res=>canvas.toBlob(res,"image/jpeg",.93));
  try{await captureLocation()}catch{}
  await processPhoto(new File([blob],"infraguard-capture.jpg",{type:"image/jpeg"}));
});

async function processPhoto(file){
  $("#permissionCard").classList.add("hidden");$("#cameraStage").classList.add("hidden");
  $("#processingPanel").classList.remove("hidden");
  const fd=new FormData();fd.append("file",file);
  try{
    const r=await api("/api/redact-image",{method:"POST",body:fd});
    state.evidenceToken=r.evidence_token;state.redactions=r.redactions;
    $("#redactedPreview").src=r.redacted_url;$("#finalEvidence").src=r.redacted_url;
    const n=r.redactions.total_redactions||0;
    $("#redactionBadge").textContent=n?`Strict privacy · ${n} detected area${n===1?"":"s"} hidden`:"Strict privacy check passed";
    $("#locationText").textContent=state.locationLabel||(
      state.lat!=null?`${state.lat.toFixed(6)}, ${state.lon.toFixed(6)}`:"Location unavailable"
    );
    stopCameraVideoOnly();
    if(r.safe_for_ai===false || !r.evidence_token){
      $("#processingTitle").textContent="Privacy verification blocked this photo";
      $("#processingText").textContent="Sensitive content may still be detectable after redaction. Please retake the photo without people, IDs, plates, documents or other private information.";
      toast("Photo blocked for privacy. Please retake it.");
      setTimeout(()=>preparePermissions(),2200);
      return;
    }
    $("#processingPanel").classList.add("hidden");$("#voiceStage").classList.remove("hidden");
    $("#voiceTitle").textContent="Allow microphone, then speak";
    $("#voiceTimer").textContent="Your browser will ask for microphone permission the first time.";
    await startVoiceAutomatically();
  }catch(e){
    $("#processingTitle").textContent="Photo processing failed";
    $("#processingText").textContent=e.message;
  }
}

function stopCameraVideoOnly(){if(!state.stream)return;state.stream.getVideoTracks().forEach(t=>t.stop())}

async function startVoiceAutomatically(){
  try{
    // Ask for microphone only when the citizen reaches the voice step.
    // This triggers the browser's first-time microphone permission prompt.
    const mic=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
    const audioTracks=mic.getAudioTracks();
    const audioStream=new MediaStream(audioTracks);
    const chunks=[];
    const preferred=["audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus"].find(x=>MediaRecorder.isTypeSupported(x));
    state.recorder=new MediaRecorder(audioStream,preferred?{mimeType:preferred}:undefined);
    state.recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
    state.recorder.onstop=async()=>{
      clearInterval(state.voiceTimer);
      state.audioBlob=new Blob(chunks,{type:state.recorder.mimeType||"audio/webm"});
      mic.getTracks().forEach(t=>t.stop());
      $("#voiceStage").classList.add("hidden");
      $("#processingPanel").classList.remove("hidden");
      $("#processingTitle").textContent="Converting voice to English…";
      $("#processingText").textContent="Sarvam detects Telugu, Hindi, English and supported Indian languages, then produces English text.";
      await transcribeAndAnalyze();
    };
    state.voiceSeconds=20;
    $("#voiceTitle").textContent="Listening…";
    $("#voiceTimer").textContent="Speak in Telugu, Hindi, English or another supported Indian language · 00:20";
    state.recorder.start();
    state.voiceTimer=setInterval(()=>{
      state.voiceSeconds--;
      $("#voiceTimer").textContent=`Speak in Telugu, Hindi, English or another supported Indian language · 00:${String(Math.max(0,state.voiceSeconds)).padStart(2,"0")}`;
      if(state.voiceSeconds<=0&&state.recorder.state==="recording")state.recorder.stop();
    },1000);
  }catch(e){
    $("#voiceTitle").textContent="Microphone permission unavailable";
    $("#voiceTimer").textContent="Allow the microphone in browser settings, or continue with photo analysis and type text manually.";
    toast(`Microphone: ${e.message}`);
    setTimeout(()=>analyzeText("","Photo analysis only — microphone unavailable"),900);
  }
}
$("#stopVoiceBtn").addEventListener("click",()=>{if(state.recorder?.state==="recording")state.recorder.stop()});

async function transcribeAndAnalyze(){
  try{
    const fd=new FormData();fd.append("file",state.audioBlob,"citizen-report.webm");
    const r=await api("/api/transcribe",{method:"POST",body:fd});
    state.transcript=r.english_text;
    await analyzeText(state.transcript,`Sarvam ${r.model} · ${r.detected_language||"auto-detected language"}`);
  }catch(e){
    toast(`Voice-to-English failed: ${e.message}`);
    await analyzeText("","Photo analysis only — voice unavailable");
  }
}

function showManualAnalysis(text){
  $("#processingPanel").classList.add("hidden");$("#analysisStage").classList.remove("hidden");
  $("#englishTranscript").textContent=text||"Voice conversion unavailable — type the complaint below.";
  $("#editText").value=text||"";
  $("#aiSource").textContent="Waiting for complaint text";
}
async function analyzeText(text,sourceNote=""){
  $("#processingPanel").classList.remove("hidden");
  $("#analysisStage").classList.add("hidden");
  $("#processingTitle").textContent="Analyzing photo + voice…";
  $("#processingText").textContent="InfraGuard combines the privacy-protected photo, English voice text, location and AI routing rules.";
  try{
    const r=await api("/api/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
      text:text||"",location_label:state.locationLabel,evidence_token:state.evidenceToken
    })});
    state.analysis=r.analysis;
    $("#processingPanel").classList.add("hidden");$("#analysisStage").classList.remove("hidden");
    $("#englishTranscript").textContent=text||"No usable voice text — photo evidence was analyzed.";
    $("#editText").value=text||"";
    const photoSource=r.photo_analysis?.source?` · Photo: ${r.photo_analysis.source}`:"";
    renderAnalysis(r.analysis,(r.live_ai?sourceNote:"Fallback analysis")+photoSource);
  }catch(e){showManualAnalysis(text);toast(e.message)}
}
function renderAnalysis(a,source){
  $("#aiIssue").textContent=a.issue_type;$("#aiSeverity").textContent=a.severity;$("#aiDepartment").textContent=a.department;
  $("#aiSummary").textContent=a.summary;$("#aiSource").textContent=source||a.ai_source;
  $("#submitBtn").disabled=false;
}
$("#editText").addEventListener("change",async()=>{
  const t=$("#editText").value.trim();if(t.length>=3){$("#submitBtn").disabled=true;await analyzeText(t,"Re-analyzed after citizen correction")}
});

$("#submitBtn").addEventListener("click",async()=>{
  if(!state.analysis||!state.evidenceToken)return;
  $("#submitBtn").disabled=true;$("#submitBtn").textContent="Submitting…";
  try{
    const r=await api("/api/submit",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
      description:$("#editText").value.trim()||state.analysis.summary||"Photo-reported infrastructure issue",transcript:state.transcript,location_label:state.locationLabel,
      latitude:state.lat,longitude:state.lon,evidence_token:state.evidenceToken,redactions:state.redactions,analysis:state.analysis
    })});
    if(r.duplicate){
      $("#submitResult").innerHTML=`<div class="notificationCard"><b>Matched existing complaint ${escapeHtml(r.report.id)}</b><p>Your report was combined with the same nearby incident. Duplicate count: ${r.report.duplicate_count}.</p></div>`;
    }else{
      $("#submitResult").innerHTML=`<div class="notificationCard"><b>Complaint Registered ✓</b><p>Case ID: ${escapeHtml(r.report.id)} · Routed to ${escapeHtml(r.report.department)}</p></div>`;
    }
    toast("Complaint submitted successfully");
    setTimeout(()=>go("myreports"),1300);
  }catch(e){toast(e.message)}
  finally{$("#submitBtn").textContent="Submit Complaint";$("#submitBtn").disabled=false}
});

function statusClass(s){return s==="Resolved"?"resolved":s==="In Progress"||s==="Assigned"?"progress":s==="Registered"?"registered":"unresolved"}
async function loadReports(){
  try{
    const {reports}=await api("/api/reports");
    $("#myReportsList").innerHTML=reports.length?reports.map(r=>`<article class="caseCard">
      <div class="caseTop"><div><span class="caseId">${escapeHtml(r.id)}</span><h3>${escapeHtml(r.issue_type)}</h3></div><span class="pill ${statusClass(r.status)}">${escapeHtml(r.status)}</span></div>
      <p>${escapeHtml(r.summary)}</p>
      <div class="caseMeta"><span>Severity<b>${escapeHtml(r.severity)}</b></span><span>Department<b>${escapeHtml(r.department)}</b></span><span>Location<b>${escapeHtml(r.location_label||"GPS attached")}</b></span><span>Community reports<b>${r.duplicate_count}</b></span></div>
      ${r.officer_notes?`<div class="notificationCard" style="margin-top:9px"><b>Officer update</b><p>${escapeHtml(r.officer_notes)}</p></div>`:""}
    </article>`).join(""):`<article class="caseCard"><h3>No reports yet</h3><p>Use Report an Issue to create your first complaint.</p></article>`;
  }catch(e){$("#myReportsList").innerHTML=`<article class="caseCard">${escapeHtml(e.message)}</article>`}
}
async function loadNotifications(){
  try{
    const {notifications}=await api("/api/notifications");
    $("#notificationList").innerHTML=notifications.length?notifications.map(n=>`<article class="notificationCard"><b>${escapeHtml(n.report_id||"InfraGuard AI")}</b><p>${escapeHtml(n.message)}</p><small>${new Date(n.created_at).toLocaleString()}</small></article>`).join(""):`<article class="notificationCard">No notifications yet.</article>`;
  }catch(e){toast(e.message)}
}

async function loadOfficer(){
  try{
    const [{reports},analytics]=await Promise.all([api("/api/reports"),api("/api/analytics")]);
    $("#officerStats").innerHTML=[
      ["Critical",analytics.severities.Critical||0],["High",analytics.severities.High||0],["Open",(analytics.total-(analytics.statuses.Resolved||0))],["Resolved",analytics.statuses.Resolved||0]
    ].map(([k,v])=>`<div class="statBox"><small>${k}</small><b>${v}</b></div>`).join("");
    $("#officerList").innerHTML=reports.sort((a,b)=>b.priority_score-a.priority_score).map(r=>`<article class="officerCase" data-id="${escapeHtml(r.id)}">
      <small>${escapeHtml(r.id)} · Priority ${r.priority_score}/100 · ${r.duplicate_count} report${r.duplicate_count===1?"":"s"}</small>
      <h3>${escapeHtml(r.issue_type)} <span class="pill">${escapeHtml(r.severity)}</span></h3>
      <p><b>Department:</b> ${escapeHtml(r.department)}<br><b>Location:</b> ${escapeHtml(r.location_label||"GPS only")}<br><b>AI:</b> ${escapeHtml(r.summary)}</p>
      <select class="offStatus">${["Registered","Assigned","In Progress","Resolved"].map(s=>`<option ${s===r.status?"selected":""}>${s}</option>`).join("")}</select>
      <textarea class="offNotes" placeholder="Officer action / resolution notes">${escapeHtml(r.officer_notes||"")}</textarea>
      <button class="primary saveOfficer">Save Case Update</button>
      <input type="file" class="resolutionFile" accept="image/*" capture="environment" style="margin-top:9px;color:#b9d0e7">
      <button class="secondary uploadResolution">Upload Resolution Evidence</button>
    </article>`).join("")||`<article class="officerCase">No live cases yet.</article>`;
  }catch(e){toast(e.message)}
}
$("#officerList").addEventListener("click",async e=>{
  const card=e.target.closest(".officerCase");if(!card)return;const id=card.dataset.id;
  if(e.target.classList.contains("saveOfficer")){
    try{await api(`/api/reports/${encodeURIComponent(id)}`,{method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify({status:$(".offStatus",card).value,officer_notes:$(".offNotes",card).value})});toast("Officer update saved");loadOfficer()}catch(err){toast(err.message)}
  }
  if(e.target.classList.contains("uploadResolution")){
    const file=$(".resolutionFile",card).files?.[0];if(!file){toast("Choose a resolution photo first");return}
    const fd=new FormData();fd.append("file",file);
    try{await api(`/api/reports/${encodeURIComponent(id)}/resolution`,{method:"POST",body:fd});toast("Resolution evidence uploaded") }catch(err){toast(err.message)}
  }
});
async function loadAnalytics(){
  try{
    const a=await api("/api/analytics");
    $("#analyticsCards").innerHTML=[["Total",a.total],["Critical",a.severities.Critical||0],["In Progress",a.statuses["In Progress"]||0],["Resolved",a.statuses.Resolved||0]].map(([k,v])=>`<div class="analyticCard"><small>${k}</small><b>${v}</b></div>`).join("");
    const max=Math.max(1,...Object.values(a.departments));
    $("#departmentBars").innerHTML=Object.entries(a.departments).map(([k,v])=>`<div class="barRow"><div class="barLabel"><span>${escapeHtml(k)}</span><b>${v}</b></div><div class="barTrack"><div class="barFill" style="width:${(v/max)*100}%"></div></div></div>`).join("")||"<p>No data yet.</p>";
  }catch(e){toast(e.message)}
}
