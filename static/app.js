const $ = (id) => document.getElementById(id);

$('sourceType').addEventListener('change', () => {
  const type = $('sourceType').value;
  $('sourceValue').value = type === 'webcam' ? '0' : '';
  $('sourceValue').placeholder = type === 'webcam' ? '0' : 'rtsp://user:password@camera/stream';
  $('sourceValueWrap').firstChild.textContent = type === 'webcam' ? 'Camera index' : 'Stream URL';
});

async function refreshStatus() {
  try {
    const s = await fetch('/api/status').then(r => r.json());
    $('fps').textContent = Number(s.fps || 0).toFixed(1);
    const b = s.ocr_backends || {};
    const parts = [s.model_loaded ? 'YOLO ready' : 'Engine unavailable'];
    if (b.paddleocr) parts.push('PaddleOCR');
    if (b.tesseract) parts.push('Tesseract');
    if (b.second_detector) parts.push('2nd detector');
    $('modelPill').textContent = parts.join(' · ');
    $('modelPill').classList.toggle('bad', !s.model_loaded);
  } catch {}
}

async function refreshEvents() {
  const rows = await fetch('/api/events?limit=60').then(r => r.json());
  $('events').innerHTML = rows.length ? rows.map(r => `
    <tr><td>${new Date(r.ts).toLocaleString()}</td><td class="plate">${escapeHtml(r.plate)}</td><td>${(r.confidence * 100).toFixed(0)}%</td><td>${escapeHtml(r.source)}</td></tr>
  `).join('') : '<tr><td colspan="4" class="muted">No detections yet.</td></tr>';
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

$('start').onclick = async () => {
  const source = $('sourceValue').value.trim() || '0';
  const res = await fetch('/api/start', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({source}) });
  if (!res.ok) {
    alert((await res.json()).error || 'Could not start stream');
    return;
  }
  $('feed').src = '/video_feed?ts=' + Date.now();
  $('emptyState').style.display = 'none';
};

$('stop').onclick = async () => {
  await fetch('/api/stop', {method:'POST'});
  $('feed').src = '';
  $('emptyState').style.display = 'grid';
};

$('refresh').onclick = refreshEvents;
setInterval(refreshStatus, 1500);
setInterval(refreshEvents, 2500);
refreshStatus();
refreshEvents();
