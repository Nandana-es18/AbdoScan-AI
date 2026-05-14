document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseLink = document.querySelector('.browse-link');
    const filePreview = document.getElementById('file-preview');
    const fileNameSpan = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file');
    const dropZoneContent = document.querySelector('.drop-zone-content');
    const submitBtn = document.getElementById('submit-btn');
    const form = document.getElementById('analyze-form');

    let currentFile = null;

    // Trigger file input click
    dropZone.addEventListener('click', (e) => {
        if (e.target !== removeFileBtn && !removeFileBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    // Handle file selection via input
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
        // Reset input value to allow selecting the same file again if needed (e.g., after removal or error)
        fileInput.value = '';
    });

    // Drag and Drop events
    const allowedExtensions = ['.dcm', '.nii', '.nii.gz', '.jpg', '.png', '.jpeg'];
    let dragCounter = 0;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    dropZone.addEventListener('dragenter', (e) => {
        dragCounter++;
        dropZone.classList.add('drag-over');
        dropZone.classList.remove('error'); // Clear error on new drag
    });

    dropZone.addEventListener('dragover', (e) => {
        e.dataTransfer.dropEffect = 'copy';
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', (e) => {
        dragCounter--;
        if (dragCounter === 0) {
            dropZone.classList.remove('drag-over');
        }
    });

    dropZone.addEventListener('drop', (e) => {
        dragCounter = 0;
        dropZone.classList.remove('drag-over');

        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (validateFile(file)) {
                currentFile = file;
                showFilePreview(currentFile.name);
                submitBtn.disabled = false;

                // Clear any previous error styling
                dropZone.classList.remove('error');
                const existingErrorMsg = dropZone.querySelector('.error-message');
                if (existingErrorMsg) existingErrorMsg.remove();
            } else {
                showError('Invalid file type. Please upload MRI or CT scans (.dcm, .nii, .nii.gz, .jpg, .png)');
            }
        }
    }

    function validateFile(file) {
        const fileName = file.name.toLowerCase();
        return allowedExtensions.some(ext => fileName.endsWith(ext));
    }

    function showError(message) {
        dropZone.classList.add('error');

        // Check if error message already exists
        let errorMsg = dropZone.querySelector('.error-message');
        if (!errorMsg) {
            errorMsg = document.createElement('p');
            errorMsg.className = 'error-message';
            // Insert after the icon wrapper
            const iconWrapper = dropZone.querySelector('.icon-wrapper');
            iconWrapper.after(errorMsg);
        }

        errorMsg.textContent = message;

        // Reset after 3 seconds
        setTimeout(() => {
            dropZone.classList.remove('error');
            if (errorMsg) errorMsg.remove();
        }, 3000);
    }

    function showFilePreview(name) {
        dropZoneContent.style.display = 'none';
        filePreview.classList.remove('hidden');
        fileNameSpan.textContent = name;
    }

    function resetFile() {
        currentFile = null;
        fileInput.value = '';
        dropZoneContent.style.display = 'block';
        filePreview.classList.add('hidden');
        submitBtn.disabled = true;
    }

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetFile();
    });

    // Modal Elements
    const modal = document.getElementById('result-modal');
    const closeModalBtn = document.getElementById('close-modal');
    const resultImage = document.getElementById('result-image');
    const resultImageContainer = document.querySelector('.result-image-container');
    const downloadLink = document.getElementById('download-link');
    const zoomInBtn = document.getElementById('zoom-in');
    const zoomOutBtn = document.getElementById('zoom-out');

    // Diagnostic Elements
    const btnLiverReport = document.getElementById('btn-liver-report');
    const btnKidneyReport = document.getElementById('btn-kidney-report');
    const diagModal = document.getElementById('diagnostic-modal');
    const closeDiagModalBtn = document.getElementById('close-diag-modal');
    const diagImage = document.getElementById('diag-image');
    const diagStatsContainer = document.getElementById('diag-stats-container');

    let currentZoom = 1.0;
    const ZOOM_STEP = 0.25;
    const MIN_ZOOM = 0.5;
    const MAX_ZOOM = 3.0;

    function updateZoom() {
        // Use transform: scale() to zoom
        resultImageContainer.style.transform = `scale(${currentZoom})`;

        // Adjust margin to ensure scrollability works nicely when zoomed in
        // (Scaling reduces layout size, so manual spacing helps scroll)
        if (currentZoom > 1) {
            resultImageContainer.style.marginTop = `${(currentZoom - 1) * 200}px`;
            resultImageContainer.style.marginBottom = `${(currentZoom - 1) * 200}px`;
        } else {
            resultImageContainer.style.marginTop = '0';
            resultImageContainer.style.marginBottom = '0';
        }
    }

    zoomInBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (currentZoom < MAX_ZOOM) {
            currentZoom += ZOOM_STEP;
            updateZoom();
        }
    });

    zoomOutBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (currentZoom > MIN_ZOOM) {
            currentZoom -= ZOOM_STEP;
            updateZoom();
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!currentFile) return;

        // Reset Zoom on new analysis
        currentZoom = 1.0;
        updateZoom();

        // Get selected modality
        const modality = document.querySelector('input[name="modality"]:checked').value;

        // UI Loading State
        const originalBtnText = submitBtn.innerHTML;
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';

        // Prepare FormData
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('modality', modality);

        try {
            console.log("Sending request to backend...");
            // Send request to backend
            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                body: formData
            });

            console.log("Response received:", response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error("Backend error:", errorText);
                throw new Error(errorText || 'Analysis failed');
            }

            // Get image blob
            const imageBlob = await response.blob();
            console.log("Image blob received:", imageBlob);
            const imageUrl = URL.createObjectURL(imageBlob);

            // Update Modal Content
            resultImage.src = imageUrl;
            downloadLink.href = imageUrl;
            downloadLink.download = `analysis_${currentFile.name}.png`;

            // Show Modal
            console.log("Showing modal...");
            modal.classList.remove('hidden');

            // Reset Form UI
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalBtnText;

        } catch (error) {
            console.error('Error during analysis:', error);
            alert(`Error: ${error.message}\nCheck console for details.`);
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalBtnText;
        }
    });

    // Modal Close Logic
    function closeModal() {
        modal.classList.add('hidden');
        resetFile();
    }

    closeModalBtn.addEventListener('click', closeModal);

    // Close on click outside
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });

    // -------------------------------------------------------------
    // DIAGNOSTIC REPORT LOGIC
    // -------------------------------------------------------------
    async function requestDiagnosticReport(organType) {
        if (!currentFile) return;

        // UI Loading State on the clicked button
        const btn = organType === 'liver' ? btnLiverReport : btnKidneyReport;
        const originalBtnText = btn.innerHTML;

        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating...';

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('organ', organType);

        // Grab modality currently selected on the page
        const modality = document.querySelector('input[name="modality"]:checked').value;
        formData.append('modality', modality);

        try {
            console.log(`Requesting ${organType} report...`);
            const response = await fetch('http://localhost:8000/diagnose', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || 'Diagnostic generation failed');
            }

            const data = await response.json();

            // 1. Update diagnostic image
            diagImage.src = data.image_url;

            // 2. Build the clinical stats UI dynamically
            let statsHtml = `
                <h3><i class="fa-solid fa-file-waveform"></i> ${data.organ_analyzed} Analysis</h3>
                <div class="diag-stats-grid">
                    <div class="stat-item">
                        <span class="stat-label">Pathology Status</span>
                        <span class="stat-value" style="color: ${data.has_tumor ? 'var(--error)' : 'var(--success)'}">
                            ${data.has_tumor ? 'SUSPICIOUS MASS DETECTED' : 'HEALTHY / NORMAL'}
                        </span>
                    </div>
            `;

            if (data.has_tumor && Object.keys(data.radiomics).length > 0) {
                for (const [key, value] of Object.entries(data.radiomics)) {
                    statsHtml += `
                        <div class="stat-item">
                            <span class="stat-label">${key}</span>
                            <span class="stat-value">${value}</span>
                        </div>
                    `;
                }
            } else if (!data.has_tumor) {
                statsHtml += `
                    <div class="stat-item">
                        <span class="stat-label">Clinical Note</span>
                        <span class="stat-value" style="font-size: 1rem; color: var(--text-muted)">
                            No clinically significant masses (>100mm³) identified.
                        </span>
                    </div>
                `;
            }

            statsHtml += `</div>`;
            diagStatsContainer.innerHTML = statsHtml;

            // 3. Show Diagnostic Modal (Hide previous if wanted, or overlap)
            diagModal.classList.remove('hidden');

        } catch (error) {
            console.error('Error during diagnostics:', error);
            alert(`Error: ${error.message}`);
        } finally {
            // Re-enable button
            btn.disabled = false;
            btn.innerHTML = originalBtnText;
        }
    }

    btnLiverReport.addEventListener('click', () => requestDiagnosticReport('liver'));
    btnKidneyReport.addEventListener('click', () => requestDiagnosticReport('kidney'));

    // Close Diagnostics Modal
    closeDiagModalBtn.addEventListener('click', () => {
        diagModal.classList.add('hidden');
    });

    diagModal.addEventListener('click', (e) => {
        if (e.target === diagModal) {
            diagModal.classList.add('hidden');
        }
    });

});
