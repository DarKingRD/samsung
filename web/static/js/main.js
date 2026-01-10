let lastResponse = null;
let selectedVariantIndex = 0;

function correctText() {
  const text = document.getElementById('input-text').value.trim();

  if (!text) {
    alert('Пожалуйста, введите текст!');
    return;
  }

  document.getElementById('loading').classList.add('show');
  document.getElementById('results').classList.remove('show');

  fetch('/api/correct', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: text })
  })
    .then(response => response.json())
    .then(data => {
      document.getElementById('loading').classList.remove('show');

      if (data.status === 'success') {
        displayResults(data);
      } else {
        alert('Ошибка: ' + data.error);
      }
    })
    .catch(error => {
      document.getElementById('loading').classList.remove('show');
      alert('Ошибка запроса: ' + error);
      console.error('Error:', error);
    });
}

function displayResults(data) {
  lastResponse = data;

  const variants = data.variants || [];
  if (!variants.length) {
    alert("Сервер не вернул variants.");
    return;
  }

  const buttons = document.getElementById('variants-buttons');
  buttons.innerHTML = '';

  variants.forEach((v, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn-secondary';
    btn.textContent = `Вариант ${i + 1} (${Math.round((v.confidence || 0) * 100)}%)`;
    btn.onclick = () => renderVariant(i);
    buttons.appendChild(btn);
  });

  renderVariant(0);
  document.getElementById('results').classList.add('show');
}

function renderVariant(idx) {
  const v = lastResponse?.variants?.[idx];
  if (!v) return;

  selectedVariantIndex = idx;

  document.getElementById('corrected-text').textContent = v.corrected_text || "";
  document.getElementById('error-count').textContent = (v.error_count ?? 0);

  const correctionsList = document.getElementById('corrections-list');
  correctionsList.innerHTML = '';

  (v.corrections || []).forEach((corr) => {
    const li = document.createElement('li');
    li.className = 'correction-item';

    li.innerHTML = `
      <div>
        <span class="correction-type ${corr.error_type}">${corr.error_type}</span>
        <span class="confidence">${((corr.confidence || 0) * 100).toFixed(0)}% уверенность</span>
      </div>
      <div class="from-to">
        <div class="from">❌ "${corr.original}"</div>
        <div class="to">✅ "${corr.corrected}"</div>
      </div>
    `;

    correctionsList.appendChild(li);
  });

  // подсветка выбранной кнопки
  document.querySelectorAll('#variants-buttons button').forEach((b, i) => {
    b.classList.toggle('btn-primary', i === idx);
    b.classList.toggle('btn-secondary', i !== idx);
  });
}

function clearAll() {
  document.getElementById('input-text').value = '';
  document.getElementById('results').classList.remove('show');
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('input-text').addEventListener('keydown', function (e) {
    if (e.ctrlKey && e.key === 'Enter') {
      correctText();
    }
  });
});
