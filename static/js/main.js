// Main JavaScript for Car Damage Detection App

document.addEventListener('DOMContentLoaded', function() {
    // Initialize file upload drag & drop functionality
    initializeFileUpload();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Handle form submissions
    handleFormSubmissions();
});

function initializeFileUpload() {
    const fileInput = document.getElementById('file');
    const uploadZone = document.querySelector('.upload-zone');
    
    // Skip if elements don't exist (i.e., on result page)
    if (!fileInput || !uploadZone) return;
    
    // Handle drag & drop events
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('border-primary');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('border-primary');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('border-primary');
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            updateFilePreview(fileInput.files[0]);
        }
    });
    
    // Handle click on upload zone
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    // Handle file selection
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            updateFilePreview(fileInput.files[0]);
        }
    });
}

function updateFilePreview(file) {
    const previewArea = document.getElementById('file-preview');
    const fileNameDisplay = document.getElementById('file-name');
    
    // Skip if elements don't exist
    if (!previewArea || !fileNameDisplay) return;
    
    // Validate file type
    if (!['image/jpeg', 'image/jpg', 'image/png'].includes(file.type)) {
        showError('Invalid file type. Please select a JPG or PNG image.');
        return;
    }
    
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File is too large. Maximum size is 16MB.');
        return;
    }
    
    // Display file name
    fileNameDisplay.textContent = file.name;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewArea.innerHTML = `<img src="${e.target.result}" class="img-fluid rounded preview-image" alt="Image preview">`;
        previewArea.classList.remove('d-none');
    };
    reader.readAsDataURL(file);
}

function handleFormSubmissions() {
    const uploadForm = document.getElementById('upload-form');
    
    // Skip if form doesn't exist
    if (!uploadForm) return;
    
    uploadForm.addEventListener('submit', function(e) {
        const fileInput = document.getElementById('file');
        
        // Validate file selected
        if (!fileInput || !fileInput.files.length) {
            e.preventDefault();
            showError('Please select an image to upload');
            return;
        }
        
        // Show loading overlay
        showLoadingOverlay();
    });
}

function showLoadingOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'spinner-overlay';
    overlay.innerHTML = `
        <div class="spinner-container">
            <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3 mb-0">Analyzing image. Please wait...</p>
        </div>
    `;
    document.body.appendChild(overlay);
}

function showError(message) {
    // Create alert if it doesn't exist
    let alertBox = document.getElementById('error-alert');
    if (!alertBox) {
        alertBox = document.createElement('div');
        alertBox.id = 'error-alert';
        alertBox.className = 'alert alert-danger alert-dismissible fade show';
        alertBox.innerHTML = `
            <span id="error-message"></span>
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        const formElement = document.querySelector('form');
        formElement.parentNode.insertBefore(alertBox, formElement);
    }
    
    // Update message and show
    document.getElementById('error-message').textContent = message;
    alertBox.classList.remove('d-none');
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        alertBox.classList.add('d-none');
    }, 5000);
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach(tooltip => {
        new bootstrap.Tooltip(tooltip);
    });
}