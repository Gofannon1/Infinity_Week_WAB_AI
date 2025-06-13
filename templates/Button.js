// Copy the text inside the element with id 'response-text'
function copyResponseText() {
  const responseTextEl = document.getElementById('response-text');
  if (!responseTextEl) {
    alert('No response text found to copy.');
    return;
  }
  const text = responseTextEl.innerText || responseTextEl.textContent;
  navigator.clipboard.writeText(text)
    .then(() => {
      alert('Text copied to clipboard!');
    })
    .catch(err => {
      alert('Error copying text: ' + err);
    });
}

// Download the text inside the element with id 'response-text' as a .txt file
function downloadResponseText(filename = 'response.txt') {
  const responseTextEl = document.getElementById('response-text');
  if (!responseTextEl) {
    alert('No response text found to download.');
    return;
  }
  const text = responseTextEl.innerText || responseTextEl.textContent;
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();

  // Cleanup
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 100);
}
