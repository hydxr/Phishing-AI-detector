// Wait for the webpage to load
document.addEventListener('DOMContentLoaded', () => {

    // Get the HTML elements
    const emailInput = document.getElementById('email-input');
    const detectBtn = document.getElementById('detect-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultContainer = document.getElementById('result-container');
    const footerTimestamp = document.querySelector('.footer-timestamp');

    // Update the clock in the footer
    function updateTimestamp() {
        footerTimestamp.textContent = new Date().toLocaleTimeString();
    }
    setInterval(updateTimestamp, 1000);
    updateTimestamp();

    // --- Add a log message to the result box ---
    function addLogMessage(message, cssClass) {
        const logEntry = document.createElement('p');
        logEntry.className = `log-message ${cssClass}`;
        logEntry.innerHTML = message; // Use innerHTML to render line breaks
        resultContainer.appendChild(logEntry);
        // Scroll to the bottom
        resultContainer.scrollTop = resultContainer.scrollHeight;
    }

    // --- "DETECT" Button Click ---
    detectBtn.addEventListener('click', async () => {
        const emailText = emailInput.value.trim();

        if (!emailText) {
            alert('Please paste an email to scan.');
            return;
        }

        // 1. Show a loading message
        resultContainer.innerHTML = ''; // Clear previous results
        addLogMessage('Scanning... Initiating threat analysis...', 'system-message');

        try {
            // 2. Send the email text to your Python server's /predict endpoint
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email_text: emailText }),
            });

            if (!response.ok) {
                throw new Error('Server returned an error.');
            }

            // 3. Get the JSON response from the server
            const result = await response.json();

            // 4. Display the result
            if (result.status === 'success') {
                const resultHtml = `
                    <span class="result-label">${result.label}</span>
                    <br>
                    CONFIDENCE: <b>${result.confidence.toFixed(2)}%</b>
                `;
                addLogMessage(resultHtml, `result-message ${result.css_class}`);
                addLogMessage('Scan complete.', 'system-message');
            } else {
                addLogMessage(`Error: ${result.message}`, 'result-message phishing');
            }

        } catch (error) {
            console.error('Error:', error);
            addLogMessage('Error: Could not connect to analysis server.', 'result-message phishing');
        }
    });

    // --- "CLEAR" Button Click ---
    clearBtn.addEventListener('click', () => {
        emailInput.value = '';
        resultContainer.innerHTML = '<p class="log-message welcome-message">System initialized. Awaiting input...</p>';
    });

});