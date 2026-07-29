// main.js - Global JavaScript functions

// Utility function to show notifications
function showNotification(message, type = 'success') {
    const colors = {
        success: '#10b981',
        error: '#ef4444',
        warning: '#f59e0b',
        info: '#3b82f6'
    };
    
    const toastHtml = `
        <div class="toast align-items-center text-white border-0 position-fixed bottom-0 end-0 m-3" 
             style="background: ${colors[type]}; z-index: 9999;" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', toastHtml);
    const toast = new bootstrap.Toast(document.querySelector('.toast:last-child'));
    toast.show();
    
    setTimeout(() => {
        document.querySelector('.toast:last-child')?.remove();
    }, 3000);
}

// Utility function to format date
function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

// Utility function to format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Utility function to get confidence level
function getConfidenceLevel(score) {
    if (score >= 0.9) return { level: 'high', text: 'High', color: '#10b981' };
    if (score >= 0.6) return { level: 'medium', text: 'Medium', color: '#f59e0b' };
    return { level: 'low', text: 'Low', color: '#ef4444' };
}

// Dark mode toggle (optional)
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
}

// Check for saved dark mode preference
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
}

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add loading state to buttons


// Page transition effects
document.querySelectorAll('a').forEach(link => {
    if (link.hostname === window.location.hostname && !link.hasAttribute('data-no-transition')) {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const href = link.getAttribute('href');
            document.body.style.opacity = '0';
            setTimeout(() => {
                window.location.href = href;
            }, 300);
        });
    }
});

// Fade in page content
document.addEventListener('DOMContentLoaded', () => {
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.3s';
        document.body.style.opacity = '1';
    }, 100);
});

// Keyboard shortcuts help
document.addEventListener('keydown', (e) => {
    // Help: Press '?' to show shortcuts
    if (e.key === '?') {
        showKeyboardShortcuts();
    }
});

function showKeyboardShortcuts() {
    const shortcuts = `
        <div class="modal fade" id="shortcutsModal" tabindex="-1">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content">
                    <div class="modal-header bg-primary text-white">
                        <h5 class="modal-title">
                            <i class="fas fa-keyboard"></i> Keyboard Shortcuts
                        </h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <ul class="list-unstyled">
                            <li><kbd>H</kbd> - Go to Home</li>
                            <li><kbd>U</kbd> - Go to Upload</li>
                            <li><kbd>D</kbd> - Go to Draw</li>
                            <li><kbd>R</kbd> - Go to History</li>
                            <li><kbd>?</kbd> - Show this help</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', shortcuts);
    const modal = new bootstrap.Modal(document.getElementById('shortcutsModal'));
    modal.show();
    
    document.getElementById('shortcutsModal').addEventListener('hidden.bs.modal', () => {
        document.getElementById('shortcutsModal').remove();
    });
}

// Handle keyboard navigation
document.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase();
    const currentPath = window.location.pathname;
    
    if (key === 'h' && currentPath !== '/') {
        window.location.href = '/';
    } else if (key === 'u' && currentPath !== '/upload') {
        window.location.href = '/upload';
    } else if (key === 'd' && currentPath !== '/draw') {
        window.location.href = '/draw';
    } else if (key === 'r' && currentPath !== '/history') {
        window.location.href = '/history';
    }
});

// Console greeting
console.log('%c✍️ Handwritten Character Recognition System', 'color: #6366f1; font-size: 16px; font-weight: bold;');
console.log('%cPowered by TensorFlow & CNN', 'color: #10b981; font-size: 12px;');
console.log('%cVisit the documentation for more info', 'color: #6b7280; font-size: 12px;');