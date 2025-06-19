// Upload functionality and UI enhancements

document.addEventListener('DOMContentLoaded', function() {
    // File upload enhancements
    initializeFileUpload();
    
    // Progress animations
    animateProgressBars();
    
    // Form validation
    setupFormValidation();
    
    // Image preview functionality
    setupImagePreview();
});

function initializeFileUpload() {
    const fileInput = document.querySelector('input[type="file"]');
    if (!fileInput) return;

    // Drag and drop functionality
    const uploadArea = fileInput.closest('.card-body');
    if (uploadArea) {
        setupDragAndDrop(uploadArea, fileInput);
    }

    // File size validation
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            validateFileSize(file);
            validateFileType(file);
        }
    });
}

function setupDragAndDrop(uploadArea, fileInput) {
    // Add visual feedback for drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('border-primary', 'bg-primary-subtle');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('border-primary', 'bg-primary-subtle');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('border-primary', 'bg-primary-subtle');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change'));
        }
    });
}

function validateFileSize(file) {
    const maxSize = 16 * 1024 * 1024; // 16MB
    if (file.size > maxSize) {
        showAlert('File size too large. Please select a file smaller than 16MB.', 'danger');
        return false;
    }
    return true;
}

function validateFileType(file) {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showAlert('Invalid file type. Please upload PNG, JPG, JPEG, TIFF, or BMP files.', 'danger');
        return false;
    }
    return true;
}

function setupImagePreview() {
    const fileInput = document.querySelector('input[type="file"]');
    if (!fileInput) return;

    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file && validateFileSize(file) && validateFileType(file)) {
            displayImagePreview(file);
        }
    });
}

function displayImagePreview(file) {
    const previewContainer = document.getElementById('imagePreview');
    const previewImage = document.getElementById('previewImage');
    const imageInfo = document.getElementById('imageInfo');

    if (previewContainer && previewImage) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            
            // Add loading animation
            previewImage.style.opacity = '0';
            previewImage.onload = function() {
                previewImage.style.transition = 'opacity 0.3s ease';
                previewImage.style.opacity = '1';
            };
            
            // Display file information
            if (imageInfo) {
                const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);
                imageInfo.innerHTML = `
                    <strong>File:</strong> ${file.name}<br>
                    <strong>Size:</strong> ${sizeInMB} MB<br>
                    <strong>Type:</strong> ${file.type}
                `;
            }
        };
        reader.readAsDataURL(file);
    }
}

function setupFormValidation() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!validateForm(form)) {
                e.preventDefault();
            }
        });
    });
}

function validateForm(form) {
    const fileInput = form.querySelector('input[type="file"]');
    if (fileInput && fileInput.required && !fileInput.files[0]) {
        showAlert('Please select an X-ray image to upload.', 'warning');
        return false;
    }
    return true;
}

function animateProgressBars() {
    const progressBars = document.querySelectorAll('.progress-bar');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const progressBar = entry.target;
                const width = progressBar.style.width || progressBar.getAttribute('data-width');
                
                if (width) {
                    progressBar.style.width = '0%';
                    setTimeout(() => {
                        progressBar.style.transition = 'width 1s ease-in-out';
                        progressBar.style.width = width;
                    }, 100);
                }
                
                observer.unobserve(progressBar);
            }
        });
    });
    
    progressBars.forEach(bar => observer.observe(bar));
}

function showAlert(message, type = 'info') {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of main content
    const mainContent = document.querySelector('.main-content') || document.body;
    mainContent.insertBefore(alertDiv, mainContent.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// API interaction functions
function uploadToAPI(file, callback) {
    const formData = new FormData();
    formData.append('xray_image', file);
    formData.append('save_to_db', 'true');
    
    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (callback) callback(data);
    })
    .catch(error => {
        console.error('API Error:', error);
        showAlert('An error occurred while processing the image.', 'danger');
    });
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    const options = { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    return new Date(dateString).toLocaleDateString(undefined, options);
}

// Export functions for use in other scripts
window.MedicalApp = {
    showAlert,
    uploadToAPI,
    formatFileSize,
    formatDate,
    validateFileSize,
    validateFileType
};
