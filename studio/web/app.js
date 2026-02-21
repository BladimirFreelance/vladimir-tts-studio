let state = { audio_id: null, text: null, index: 0, total: 0, done: false };
let mediaRecorder = null;
let chunks = [];
let stream = null;
let startedAt = null;
let timerHandle = null;
let recordedBlob = null;

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.detail || JSON.stringify(data));
  return data;
}

function renderPrompt() {
  document.getElementById('line').innerText = state.done ? 'Готово! Все фразы записаны.' : (state.text || 'Сначала подготовьте датасет');
  document.getElementById('progress').innerText = `${state.index || 0}/${state.total || 0}`;
}

async function fetchNext() {
  state = await api('/api/next');
  renderPrompt();
}

async function refreshStatus() {
  try {
    const status = await api('/api/status');
    document.getElementById('projectInfo').innerText = `Проект: ${status.project} (${status.project_dir})`;
    document.getElementById('status').innerText = `Подготовлен: ${status.prepared ? 'да' : 'нет'} | Записано: ${status.recorded}/${status.total} | Задача: ${status.task.name || '-'} (${status.task.status})`;
    const result = status.task.error || JSON.stringify(status.task.last_result || {}, null, 2);
    document.getElementById('taskResult').innerText = result;
  } catch (err) {
    document.getElementById('status').innerText = `Ошибка обновления статуса: ${err.message}`;
  }
}

function fillDatalist(elementId, options) {
  const list = document.getElementById(elementId);
  list.innerHTML = '';
  for (const option of options || []) {
    const node = document.createElement('option');
    node.value = option;
    list.appendChild(node);
  }
}

async function refreshModelOptions() {
  const models = await api('/api/models');
  fillDatalist('baseCkptOptions', models.ckpt_options || []);
  fillDatalist('resumeCkptOptions', models.ckpt_options || []);
  fillDatalist('exportCkptOptions', models.ckpt_options || []);
  fillDatalist('onnxModelOptions', models.onnx_options || []);

  const base = document.getElementById('baseCkpt');
  if (!base.value.trim()) {
    base.value = models.default_base_ckpt || '';
  }

  const resume = document.getElementById('resumeCkpt');
  if (!resume.value.trim() && models.default_resume_ckpt) {
    resume.value = models.default_resume_ckpt;
  }

  const exportCkpt = document.getElementById('exportCkpt');
  if (!exportCkpt.value.trim() && models.default_export_ckpt) {
    exportCkpt.value = models.default_export_ckpt;
  }

  const testModel = document.getElementById('testModel');
  if (!testModel.value.trim() && models.default_test_onnx) {
    testModel.value = models.default_test_onnx;
  }
}

async function runAction(path, payload) {
  try {
    await api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    await refreshStatus();
    await refreshModelOptions();
  } catch (err) {
    alert(err.message);
  }
}

async function uploadPrepareText(file) {
  const form = new FormData();
  form.append('file', file);
  const resp = await fetch('/api/prepare/upload', { method: 'POST', body: form });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.detail || JSON.stringify(data));
  return data;
}

async function initAudio() {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  source.connect(analyser);
  const data = new Uint8Array(analyser.frequencyBinCount);
  setInterval(() => {
    analyser.getByteFrequencyData(data);
    const avg = data.reduce((a, b) => a + b, 0) / data.length;
    document.getElementById('level').style.width = `${Math.min(100, avg / 2)}%`;
  }, 120);
}

function startRecording() {
  chunks = [];
  recordedBlob = null;
  mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
  mediaRecorder.ondataavailable = e => chunks.push(e.data);
  mediaRecorder.onstop = () => {
    if (!chunks.length) return;
    recordedBlob = new Blob(chunks, { type: 'audio/webm' });
    document.getElementById('preview').src = URL.createObjectURL(recordedBlob);
    document.getElementById('preview').load();
  };
  mediaRecorder.start();
  startedAt = Date.now();
  timerHandle = setInterval(() => {
    const sec = Math.floor((Date.now() - startedAt) / 1000);
    const mm = String(Math.floor(sec / 60)).padStart(2, '0');
    const ss = String(sec % 60).padStart(2, '0');
    document.getElementById('timer').innerText = `${mm}:${ss}`;
  }, 250);
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    clearInterval(timerHandle);
  }
}

async function saveRecording() {
  if (!recordedBlob || !state.audio_id) {
    alert('Сначала запишите и остановите фразу, затем сохраните её.');
    return;
  }

  const wavBlob = await convertToWav(recordedBlob);
  const form = new FormData();
  form.append('audio_id', state.audio_id);
  form.append('text', state.text);
  form.append('file', wavBlob, `${state.audio_id}.wav`);
  const resp = await fetch('/api/save', { method: 'POST', body: form });
  const payload = await resp.json();
  if (!resp.ok) {
    throw new Error(payload.detail || JSON.stringify(payload));
  }

  recordedBlob = null;
  chunks = [];
  document.getElementById('preview').removeAttribute('src');
  document.getElementById('preview').load();
  document.getElementById('timer').innerText = '00:00';

  await fetchNext();
  await refreshStatus();
}

async function convertToWav(blob) {
  const audioCtx = new AudioContext({ sampleRate: 22050 });
  const arr = await blob.arrayBuffer();
  const decoded = await audioCtx.decodeAudioData(arr);
  const channel = decoded.getChannelData(0);
  const wav = encodeWAV(channel, decoded.sampleRate);
  return new Blob([wav], { type: 'audio/wav' });
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  function writeString(offset, str) { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); }
  let offset = 0;
  writeString(offset, 'RIFF'); offset += 4;
  view.setUint32(offset, 36 + samples.length * 2, true); offset += 4;
  writeString(offset, 'WAVE'); offset += 4;
  writeString(offset, 'fmt '); offset += 4;
  view.setUint32(offset, 16, true); offset += 4;
  view.setUint16(offset, 1, true); offset += 2;
  view.setUint16(offset, 1, true); offset += 2;
  view.setUint32(offset, sampleRate, true); offset += 4;
  view.setUint32(offset, sampleRate * 2, true); offset += 4;
  view.setUint16(offset, 2, true); offset += 2;
  view.setUint16(offset, 16, true); offset += 2;
  writeString(offset, 'data'); offset += 4;
  view.setUint32(offset, samples.length * 2, true); offset += 4;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

document.getElementById('prepareBtn').onclick = () => runAction('/api/prepare', { text_path: document.getElementById('prepareTextPath').value });
document.getElementById('pickTextFileBtn').onclick = () => document.getElementById('prepareTextFile').click();
document.getElementById('prepareTextFile').onchange = async (event) => {
  const [file] = event.target.files || [];
  if (!file) return;
  try {
    const result = await uploadPrepareText(file);
    document.getElementById('prepareTextPath').value = result.text_path;
    await refreshStatus();
  } catch (err) {
    alert(err.message);
  } finally {
    event.target.value = '';
  }
};
document.getElementById('trainBtn').onclick = () => runAction('/api/train', {
  epochs: Number(document.getElementById('epochs').value),
  base_ckpt: document.getElementById('baseCkpt').value,
  resume_ckpt: document.getElementById('resumeCkpt').value,
});
document.getElementById('exportBtn').onclick = () => runAction('/api/export', { ckpt: document.getElementById('exportCkpt').value });
document.getElementById('testBtn').onclick = () => runAction('/api/test', {
  model: document.getElementById('testModel').value,
  text: document.getElementById('testText').value,
  out: document.getElementById('testOut').value,
});
document.getElementById('doctorBtn').onclick = () => runAction('/api/doctor', { auto_fix: document.getElementById('doctorFix').checked });

document.getElementById('start').onclick = startRecording;
document.getElementById('stop').onclick = stopRecording;
document.getElementById('save').onclick = async () => {
  try {
    await saveRecording();
  } catch (err) {
    alert(err.message);
  }
};
document.getElementById('repeat').onclick = async () => { await api('/api/repeat', { method: 'POST' }); renderPrompt(); };
document.getElementById('bad').onclick = async () => {
  await api('/api/bad', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ audio_id: state.audio_id, text: state.text }),
  });
  await fetchNext();
  await refreshStatus();
};
document.getElementById('back').onclick = async () => { state = await api('/api/back',{method:'POST'}); renderPrompt(); await refreshStatus(); };

(async () => {
  await initAudio();
  await fetchNext();
  await refreshStatus();
  await refreshModelOptions();
  setInterval(refreshStatus, 2000);
})();
