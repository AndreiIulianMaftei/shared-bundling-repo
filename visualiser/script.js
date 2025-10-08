class ImageVisualizer {
    constructor() {
        this.images = [];
        this.filteredImages = [];
        this.currentImageIndex = 0;
        this.currentView = 'grid'; // 'grid' or 'slider'
        
        this.initializeElements();
        this.attachEventListeners();
        this.updateUI();
    }

    initializeElements() {
        // View containers
        this.gridView = document.getElementById('gridView');
        this.sliderView = document.getElementById('sliderView');
        
        // Controls
        this.gridViewBtn = document.getElementById('gridViewBtn');
        this.sliderViewBtn = document.getElementById('sliderViewBtn');
        this.imageFolder = document.getElementById('imageFolder');
        this.imageFilter = document.getElementById('imageFilter');
        this.sortBy = document.getElementById('sortBy');
        
        // Grid elements
        this.imageGrid = document.getElementById('imageGrid');
        
        // Slider elements
        this.currentImage = document.getElementById('currentImage');
        this.currentImageTitle = document.getElementById('currentImageTitle');
        this.currentImageDetails = document.getElementById('currentImageDetails');
        this.imageSlider = document.getElementById('imageSlider');
        this.prevBtn = document.getElementById('prevBtn');
        this.nextBtn = document.getElementById('nextBtn');
        this.thumbnailContainer = document.getElementById('thumbnailContainer');
        this.sliderCurrent = document.getElementById('sliderCurrent');
        this.sliderMin = document.getElementById('sliderMin');
        this.sliderMax = document.getElementById('sliderMax');
        
        // UI elements
        this.imageCount = document.getElementById('imageCount');
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.emptyState = document.getElementById('emptyState');
        
        // Modal elements
        this.imageModal = new bootstrap.Modal(document.getElementById('imageModal'));
        this.modalImage = document.getElementById('modalImage');
        this.modalImageTitle = document.getElementById('modalImageTitle');
        this.modalImageDetails = document.getElementById('modalImageDetails');
    }

    attachEventListeners() {
        // View toggle buttons
        this.gridViewBtn.addEventListener('click', () => this.switchView('grid'));
        this.sliderViewBtn.addEventListener('click', () => this.switchView('slider'));
        
        // File input
        this.imageFolder.addEventListener('change', (e) => this.handleFileSelection(e));
        
        // Filter and sort
        this.imageFilter.addEventListener('input', () => this.filterImages());
        this.sortBy.addEventListener('change', () => this.sortImages());
        
        // Slider controls
        this.prevBtn.addEventListener('click', () => this.previousImage());
        this.nextBtn.addEventListener('click', () => this.nextImage());
        this.imageSlider.addEventListener('input', (e) => this.setCurrentImage(parseInt(e.target.value)));
        this.currentImage.addEventListener('click', () => this.showImageModal());
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
    }

    async handleFileSelection(event) {
        const files = Array.from(event.target.files);
        const imageFiles = files.filter(file => file.type.startsWith('image/'));
        
        if (imageFiles.length === 0) {
            this.showMessage('No image files found in the selected directory.', 'warning');
            return;
        }

        this.showLoading(true);
        this.images = [];

        try {
            for (const file of imageFiles) {
                const imageData = await this.processImageFile(file);
                this.images.push(imageData);
            }

            this.filteredImages = [...this.images];
            this.sortImages();
            this.currentImageIndex = 0;
            this.updateUI();
            this.showMessage(`Loaded ${this.images.length} images successfully.`, 'success');
        } catch (error) {
            console.error('Error processing images:', error);
            this.showMessage('Error loading images. Please try again.', 'danger');
        } finally {
            this.showLoading(false);
        }
    }

    async processImageFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    resolve({
                        name: file.name,
                        path: file.webkitRelativePath || file.name,
                        url: e.target.result,
                        size: file.size,
                        lastModified: file.lastModified,
                        dimensions: {
                            width: img.width,
                            height: img.height
                        }
                    });
                };
                img.onerror = reject;
                img.src = e.target.result;
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    filterImages() {
        const filterText = this.imageFilter.value.toLowerCase();
        this.filteredImages = this.images.filter(image => 
            image.name.toLowerCase().includes(filterText) ||
            image.path.toLowerCase().includes(filterText)
        );
        
        this.currentImageIndex = 0;
        this.updateUI();
    }

    sortImages() {
        const sortBy = this.sortBy.value;
        
        this.filteredImages.sort((a, b) => {
            switch (sortBy) {
                case 'name':
                    return a.name.localeCompare(b.name);
                case 'date':
                    return b.lastModified - a.lastModified;
                case 'size':
                    return b.size - a.size;
                default:
                    return 0;
            }
        });
        
        this.updateUI();
    }

    switchView(view) {
        this.currentView = view;
        
        // Update button states
        this.gridViewBtn.classList.toggle('active', view === 'grid');
        this.sliderViewBtn.classList.toggle('active', view === 'slider');
        
        // Toggle view containers
        this.gridView.classList.toggle('d-none', view !== 'grid');
        this.sliderView.classList.toggle('d-none', view !== 'slider');
        
        this.updateUI();
    }

    updateUI() {
        this.updateImageCount();
        
        if (this.filteredImages.length === 0) {
            this.showEmptyState(true);
            return;
        }
        
        this.showEmptyState(false);
        
        if (this.currentView === 'grid') {
            this.updateGridView();
        } else {
            this.updateSliderView();
        }
    }

    updateImageCount() {
        const count = this.filteredImages.length;
        this.imageCount.textContent = `${count} image${count !== 1 ? 's' : ''} loaded`;
    }

    updateGridView() {
        this.imageGrid.innerHTML = '';
        
        this.filteredImages.forEach((image, index) => {
            const col = document.createElement('div');
            col.className = 'col-lg-4 col-md-6 col-sm-12 grid-item';
            
            col.innerHTML = `
                <div class="image-card fade-in" onclick="visualizer.openImageModal(${index})">
                    <img src="${image.url}" alt="${image.name}" loading="lazy">
                </div>
            `;
            
            this.imageGrid.appendChild(col);
        });
    }

    updateSliderView() {
        if (this.filteredImages.length === 0) return;
        
        // Ensure current index is valid
        this.currentImageIndex = Math.max(0, Math.min(this.currentImageIndex, this.filteredImages.length - 1));
        
        const currentImage = this.filteredImages[this.currentImageIndex];
        
        // Update current image
        this.currentImage.src = currentImage.url;
        this.currentImage.alt = currentImage.name;
        this.currentImageTitle.textContent = currentImage.name;
        this.currentImageDetails.textContent = 
            `${currentImage.dimensions.width} × ${currentImage.dimensions.height} px • ${this.formatFileSize(currentImage.size)}`;
        
        // Update slider
        this.imageSlider.max = this.filteredImages.length - 1;
        this.imageSlider.value = this.currentImageIndex;
        this.imageSlider.disabled = this.filteredImages.length <= 1;
        
        // Update slider labels
        this.sliderMin.textContent = '1';
        this.sliderMax.textContent = this.filteredImages.length.toString();
        this.sliderCurrent.textContent = `${this.currentImageIndex + 1} of ${this.filteredImages.length}`;
        
        // Update navigation buttons
        this.prevBtn.disabled = this.currentImageIndex === 0;
        this.nextBtn.disabled = this.currentImageIndex === this.filteredImages.length - 1;
        
        // Update thumbnails
        this.updateThumbnails();
    }

    updateThumbnails() {
        this.thumbnailContainer.innerHTML = '';
        
        this.filteredImages.forEach((image, index) => {
            const thumbnail = document.createElement('img');
            thumbnail.src = image.url;
            thumbnail.alt = image.name;
            thumbnail.className = `thumbnail ${index === this.currentImageIndex ? 'active' : ''}`;
            thumbnail.title = image.name;
            thumbnail.onclick = () => this.setCurrentImage(index);
            
            this.thumbnailContainer.appendChild(thumbnail);
        });
        
        // Scroll to active thumbnail
        const activeThumbnail = this.thumbnailContainer.querySelector('.thumbnail.active');
        if (activeThumbnail) {
            activeThumbnail.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
        }
    }

    setCurrentImage(index) {
        if (index >= 0 && index < this.filteredImages.length) {
            this.currentImageIndex = index;
            this.updateSliderView();
        }
    }

    previousImage() {
        if (this.currentImageIndex > 0) {
            this.setCurrentImage(this.currentImageIndex - 1);
        }
    }

    nextImage() {
        if (this.currentImageIndex < this.filteredImages.length - 1) {
            this.setCurrentImage(this.currentImageIndex + 1);
        }
    }

    openImageModal(index) {
        if (this.currentView === 'grid') {
            this.currentImageIndex = index;
        }
        this.showImageModal();
    }

    showImageModal() {
        if (this.filteredImages.length === 0) return;
        
        const image = this.filteredImages[this.currentImageIndex];
        this.modalImage.src = image.url;
        this.modalImageTitle.textContent = image.name;
        this.modalImageDetails.textContent = 
            `${image.dimensions.width} × ${image.dimensions.height} pixels • ${this.formatFileSize(image.size)} • ${new Date(image.lastModified).toLocaleString()}`;
        
        this.imageModal.show();
    }

    handleKeyboard(event) {
        if (this.currentView !== 'slider' || this.filteredImages.length === 0) return;
        
        switch (event.key) {
            case 'ArrowLeft':
                event.preventDefault();
                this.previousImage();
                break;
            case 'ArrowRight':
                event.preventDefault();
                this.nextImage();
                break;
            case 'Home':
                event.preventDefault();
                this.setCurrentImage(0);
                break;
            case 'End':
                event.preventDefault();
                this.setCurrentImage(this.filteredImages.length - 1);
                break;
            case 'Escape':
                if (document.querySelector('.modal.show')) {
                    this.imageModal.hide();
                }
                break;
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showLoading(show) {
        this.loadingSpinner.classList.toggle('d-none', !show);
        this.gridView.classList.toggle('d-none', show || this.currentView !== 'grid');
        this.sliderView.classList.toggle('d-none', show || this.currentView !== 'slider');
    }

    showEmptyState(show) {
        this.emptyState.classList.toggle('d-none', !show);
        this.gridView.classList.toggle('d-none', show || this.currentView !== 'grid');
        this.sliderView.classList.toggle('d-none', show || this.currentView !== 'slider');
    }

    showMessage(message, type = 'info') {
        // Create toast notification
        const toastContainer = document.getElementById('toastContainer') || this.createToastContainer();
        
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // Remove toast element after it's hidden
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }

    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
        return container;
    }
}

// Initialize the visualizer when the page loads
let visualizer;
document.addEventListener('DOMContentLoaded', () => {
    visualizer = new ImageVisualizer();
});

// Add some sample data loader for testing (remove in production)
function loadSampleImages() {
    // This function can be used to load sample images for testing
    // In a real scenario, users will upload their own images
    console.log('To test the visualizer, please select a folder containing images using the file input.');
}