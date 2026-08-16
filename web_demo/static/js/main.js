/* =====================================================
   EmotionAI — Frontend Logic
   ===================================================== */

// Emoji & colour maps
const emotionEmojis = {
    'Angry':    '😠',
    'Disgust':  '🤢',
    'Fear':     '😨',
    'Happy':    '😊',
    'Neutral':  '😐',
    'Sad':      '😢',
    'Surprise': '😲',
};

const emotionColors = {
    'Angry':    '#e74c3c',
    'Disgust':  '#9b59b6',
    'Fear':     '#e67e22',
    'Happy':    '#2ecc71',
    'Neutral':  '#3498db',
    'Sad':      '#1abc9c',
    'Surprise': '#f1c40f',
};

// DOM references
const dropzone          = document.getElementById('dropzone');
const dropzoneContent   = document.getElementById('dropzoneContent');
const fileInput         = document.getElementById('fileInput');
const previewArea       = document.getElementById('previewArea');
const originalImage     = document.getElementById('originalImage');
const preprocessedImage = document.getElementById('preprocessedImage');
const analyzeBtn        = document.getElementById('analyzeBtn');
const resetBtn          = document.getElementById('resetBtn');
const loadingSpinner    = document.getElementById('loadingSpinner');
const errorMsg          = document.getElementById('errorMsg');
const errorText         = document.getElementById('errorText');
const resultsSection    = document.getElementById('results');
const resultEmoji       = document.getElementById('resultEmoji');
const resultEmotion     = document.getElementById('resultEmotion');
const resultConfidence  = document.getElementById('resultConfidence');
const confidenceBars    = document.getElementById('confidenceBars');

let selectedFile = null;

// =====================================================
// Drag & Drop
// =====================================================
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

// =====================================================
// File handling
// =====================================================
function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file (JPG, PNG, WEBP).');
        return;
    }
    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImage.src = e.target.result;
        preprocessedImage.src = '';  // will be set after prediction
        preprocessedImage.alt = 'Will appear after analysis';
    };
    reader.readAsDataURL(file);

    hideError();
    hideResults();
    dropzone.style.display = 'none';
    previewArea.style.display = 'block';
    previewArea.style.animation = 'none';
    // Trigger reflow for re-animation
    void previewArea.offsetWidth;
    previewArea.style.animation = 'fadeInUp 0.5s ease';
}

// =====================================================
// Reset
// =====================================================
resetBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    originalImage.src = '';
    preprocessedImage.src = '';
    previewArea.style.display = 'none';
    dropzone.style.display = '';
    hideResults();
    hideError();
});

// =====================================================
// Analyze
// =====================================================
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    loadingSpinner.style.display = 'block';
    hideError();
    hideResults();

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!data.success) {
            showError(data.error || 'Unknown error occurred.');
            return;
        }

        // Set preprocessed image
        if (data.preprocessed_image) {
            preprocessedImage.src = data.preprocessed_image;
        }

        // Display results
        displayResults(data);

    } catch (err) {
        showError('Failed to connect to the server. Make sure app.py is running.');
        console.error(err);
    } finally {
        analyzeBtn.disabled = false;
        loadingSpinner.style.display = 'none';
    }
});

// =====================================================
// Display results
// =====================================================
function displayResults(data) {
    const { predicted_emotion, predicted_confidence, predictions } = data;

    // Primary result
    resultEmoji.textContent = emotionEmojis[predicted_emotion] || '🔍';
    resultEmotion.textContent = predicted_emotion;
    resultConfidence.textContent = `${predicted_confidence.toFixed(1)}% confidence`;

    // Build bars
    confidenceBars.innerHTML = '';
    predictions.forEach((p, i) => {
        const row = document.createElement('div');
        row.classList.add('bar-row');
        row.style.animationDelay = `${i * 0.08}s`;

        const color = emotionColors[p.emotion] || '#888';

        row.innerHTML = `
            <span class="bar-row__label">
                <span>${emotionEmojis[p.emotion] || ''}</span>
                ${p.emotion}
            </span>
            <div class="bar-row__track">
                <div class="bar-row__fill" style="background:${color};" data-width="${p.confidence}"></div>
            </div>
            <span class="bar-row__value">${p.confidence.toFixed(1)}%</span>
        `;

        confidenceBars.appendChild(row);
    });

    // Show section
    resultsSection.style.display = 'block';

    // Animate bars after a short delay so the DOM is ready
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            document.querySelectorAll('.bar-row__fill').forEach((bar) => {
                bar.style.width = bar.dataset.width + '%';
            });
        });
    });

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);
}

// =====================================================
// Helpers
// =====================================================
function showError(msg) {
    errorText.textContent = msg;
    errorMsg.style.display = 'flex';
}

function hideError() {
    errorMsg.style.display = 'none';
}

function hideResults() {
    resultsSection.style.display = 'none';
}
