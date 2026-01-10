
        function correctText() {
            const text = document.getElementById('input-text').value.trim();
            
            if (!text) {
                alert('Пожалуйста, введите текст!');
                return;
            }
            
            // Показываем загрузку
            document.getElementById('loading').classList.add('show');
            document.getElementById('results').classList.remove('show');
            
            // Отправляем запрос
            fetch('/api/correct', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
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
            // Показываем исправленный текст
            document.getElementById('corrected-text').textContent = data.corrected_text;
            document.getElementById('error-count').textContent = data.error_count;
            
            // Показываем список ошибок
            const correctionsList = document.getElementById('corrections-list');
            correctionsList.innerHTML = '';
            
            data.corrections.forEach((corr, idx) => {
                const li = document.createElement('li');
                li.className = 'correction-item';
                
                li.innerHTML = `
                    <div>
                        <span class="correction-type ${corr.error_type}">${corr.error_type}</span>
                        <span class="confidence">${(corr.confidence * 100).toFixed(0)}% уверенность</span>
                    </div>
                    <div class="from-to">
                        <div class="from">❌ "${corr.original}"</div>
                        <div class="to">✅ "${corr.corrected}"</div>
                    </div>
                `;
                
                correctionsList.appendChild(li);
            });
            
            document.getElementById('results').classList.add('show');
        }
        
        function clearAll() {
            document.getElementById('input-text').value = '';
            document.getElementById('results').classList.remove('show');
        }
        
        // Позволяем исправлять по Ctrl+Enter
        document.getElementById('input-text').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                correctText();
            }
        });