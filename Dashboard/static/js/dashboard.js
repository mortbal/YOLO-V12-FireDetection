class TrainingDashboard {
    constructor() {
        this.isTraining = false;
        this.isTesting = false;
        this.socket = null;
        this.unfinishedTrainingInfo = null;
        
        this.initializeElements();
        this.attachEventListeners();
        this.handleMobileResponsive();
        this.initializeWebSocket();
        this.loadAllModels();
        this.loadTrainedModels();
        this.checkForUnfinishedTraining();
        this.initializeLiveDetection();
        this.loadAppState();
    }

    initializeElements() {
        // Sidebar elements
        this.sidebar = document.getElementById('sidebar');
        this.sidebarToggle = document.getElementById('sidebar-toggle');
        this.navItems = document.querySelectorAll('.nav-item');
        this.tabContents = document.querySelectorAll('.tab-content');
        
        // Training configuration elements
        this.modelSelect = document.getElementById('model-select');
        this.epochsInput = document.getElementById('epochs');
        this.startButton = document.getElementById('start-training');
        this.debugLoggingCheckbox = document.getElementById('debug-logging-checkbox');
        this.statusOverlay = document.getElementById('training-status-overlay');
        this.statusText = document.getElementById('training-status-text');
        
        // Progress elements
        this.currentEpochDisplay = document.getElementById('current-epoch');
        this.totalEpochsDisplay = document.getElementById('total-epochs');
        this.epochProgressFill = document.getElementById('epoch-progress-fill');
        this.epochProgressPercentage = document.getElementById('epoch-progress-percentage');
        this.totalProgressFill = document.getElementById('total-progress-fill');
        this.totalProgressPercentage = document.getElementById('total-progress-percentage');
        this.gpuMemoryDisplay = document.getElementById('gpu-memory');
        this.gpuProgressFill = document.getElementById('gpu-progress-fill');
        this.gpuPercentageDisplay = document.getElementById('gpu-percentage');
        this.epochElapsedTime = document.getElementById('epoch-elapsed-time');
        this.estimatedRemainingTime = document.getElementById('estimated-remaining-time');
        this.totalElapsedTime = document.getElementById('total-elapsed-time');
        this.totalRemainingTime = document.getElementById('total-remaining-time');
        
// Training state
        this.trainingStartTime = null;
        this.epochDurations = [];
        this.currentEpoch = 0;
        this.totalTrainingTimer = null;
        this.trainingInitialized = false;
        this.lastGpuMemory = null;
        
        this.testFilesFolder = document.getElementById('test-files-folder');
        this.resultFilesFolder = document.getElementById('result-files-folder');
        this.browseTestFolderButton = document.getElementById('browse-test-folder');
        this.browseResultFolderButton = document.getElementById('browse-result-folder');
        this.startDetectionButton = document.getElementById('start-detection');
        this.detectionStatusDisplay = document.getElementById('detection-status');
        this.detectionResultsDisplay = document.getElementById('detection-results');
        
        // Live detection elements
        this.cameraSelect = document.getElementById('camera-select');
        this.liveModelSelect = document.getElementById('live-model-select');
        this.startLiveDetectionButton = document.getElementById('start-live-detection');
        this.cameraVideo = document.getElementById('camera-video');
        this.processedFrame = document.getElementById('processed-frame');
        this.detectionCanvas = document.getElementById('detection-canvas');
        this.cameraPlaceholder = document.getElementById('camera-placeholder');
        this.cameraError = document.getElementById('camera-error');
        this.feedStatus = document.getElementById('feed-status');
        this.errorMessage = document.getElementById('error-message');
        this.retryButton = document.getElementById('retry-camera');
        this.fireCountDisplay = document.getElementById('fire-count');
        this.smokeCountDisplay = document.getElementById('smoke-count');
        this.liveDetectionLog = document.getElementById('live-detection-log');
        
        // Resume training dialog elements
        this.resumeDialog = document.getElementById('resume-training-dialog');
        this.resumeProgressFill = document.getElementById('resume-progress-fill');
        this.resumeProgressText = document.getElementById('resume-progress-text');
        this.resumeCurrentEpoch = document.getElementById('resume-current-epoch');
        this.resumeTotalEpochs = document.getElementById('resume-total-epochs');
        this.resumeCheckpointType = document.getElementById('resume-checkpoint-type');
        this.resumeAdditionalEpochs = document.getElementById('resume-additional-epochs');
        this.resumeTrainingBtn = document.getElementById('resume-training-btn');
        this.startNewTrainingBtn = document.getElementById('start-new-training-btn');
        this.cancelResumeBtn = document.getElementById('cancel-resume-btn');
        this.closeResumeDialog = document.getElementById('close-resume-dialog');
        
        // Test tab elements
        this.trainedModelSelect = document.getElementById('trained-model-select');
        
    }

    attachEventListeners() {
        // Sidebar toggle functionality
        this.sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        
        // Navigation item clicks
        this.navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                const tabName = item.dataset.tab;
                if (tabName) {
                    this.switchTab(tabName);
                }
            });
        });
        
        if (this.startButton) {
            this.startButton.addEventListener('click', () => this.handleTrainingButton());
        }
        
        // Debug logging checkbox
        if (this.debugLoggingCheckbox) {
            this.debugLoggingCheckbox.addEventListener('change', () => this.toggleDebugLogging());
        }
        
        
        this.browseTestFolderButton.addEventListener('click', () => this.browseFolder('test'));
        this.browseResultFolderButton.addEventListener('click', () => this.browseFolder('result'));
        this.startDetectionButton.addEventListener('click', () => this.startDetection());
        
        // Live detection event listeners
        if (this.startLiveDetectionButton) {
            this.startLiveDetectionButton.addEventListener('click', () => this.toggleLiveDetection());
        }
        if (this.retryButton) {
            this.retryButton.addEventListener('click', () => this.initializeCamera());
        }
        
        // Resume dialog event listeners
        if (this.resumeTrainingBtn) {
            this.resumeTrainingBtn.addEventListener('click', () => this.resumeTraining());
        }
        if (this.startNewTrainingBtn) {
            this.startNewTrainingBtn.addEventListener('click', () => this.startNewTraining());
        }
        if (this.cancelResumeBtn) {
            this.cancelResumeBtn.addEventListener('click', () => this.hideResumeDialog());
        }
        if (this.closeResumeDialog) {
            this.closeResumeDialog.addEventListener('click', () => this.hideResumeDialog());
        }
        
    }

    startTraining() {
        if (this.isTraining) return;

        const config = {
            model: this.modelSelect.value,
            epochs: parseInt(this.epochsInput.value)
        };

        if (!this.validateConfig(config)) return;

        this.isTraining = true;
        this.initializeTrainingTimers();
        this.updateUI();
        this.startRealTraining(config);
    }
    
    handleTrainingButton() {
        if (this.isTraining) {
            this.cancelTraining();
        } else {
            this.startTraining();
        }
    }
    
    async cancelTraining() {
        if (confirm('Are you sure you want to cancel the training?')) {
            try {
                // Call backend to stop training
                const response = await fetch('/cancel_training', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ save_checkpoint: false })
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.message || 'Failed to cancel training on backend');
                }
                
                const result = await response.json();
                
                // Stop the training locally
                this.isTraining = false;
                this.stopTotalElapsedTimer();
                this.updateUI();
                
                this.statusText.textContent = 'Training cancelled';
                
            } catch (error) {
                console.error('Error cancelling training:', error);
                alert('Failed to cancel training: ' + error.message);
            }
        }
    }
    
    toggleDebugLogging() {
        const enabled = this.debugLoggingCheckbox.checked;
        
        fetch('/set_debug_logging', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ enabled: enabled })
        })
        .then(response => response.json())
        .then(data => {
            console.log(enabled ? 'Debug logging enabled' : 'Debug logging disabled');
        })
        .catch(error => {
            console.error('Error setting debug logging:', error);
        });
    }

    initializeTrainingTimers() {
        this.trainingStartTime = Date.now();
        this.epochDurations = [];
        this.trainingInitialized = false;
        this.lastGpuMemory = null;
        this.totalElapsedTimer = null;
        this.lastEpochTime = null;
        this.estimatedTotalRemaining = null;
        
        this.startTotalElapsedTimer();
        
        this.epochElapsedTime.textContent = '-';
        this.estimatedRemainingTime.textContent = '-';
        this.totalElapsedTime.textContent = '-';
        this.totalRemainingTime.textContent = 'Calculating...';
        this.currentEpochDisplay.textContent = '-';
        this.totalEpochsDisplay.textContent = '-';
        this.epochProgressFill.style.width = '0%';
        this.epochProgressPercentage.textContent = '0%';
        this.totalProgressFill.style.width = '0%';
        this.totalProgressPercentage.textContent = '0%';
        this.gpuMemoryDisplay.textContent = '-';
        if (this.gpuProgressFill) this.updateGpuBar(0);
        this.gpuPercentageDisplay.textContent = '0%';
    }


    validateConfig(config) {
        if (!config.model || config.model === "") {
            alert('Please select a model to train');
            return false;
        }
        if (config.epochs < 1 || config.epochs > 1000) {
            alert('Please enter a valid number of epochs (1-1000)');
            return false;
        }
        return true;
    }

    updateUI() {
        // Change button text and style based on training status
        if (this.isTraining) {
            this.startButton.textContent = 'Cancel Training';
            this.startButton.className = 'btn-cancel'; // Red cancel button
            this.startButton.disabled = false;
            
            // Disable model selection and epochs input during training
            this.modelSelect.disabled = true;
            this.epochsInput.disabled = true;
            
            this.showStatusOverlay('Preparing for train : Loading YOLO...');
        } else {
            this.startButton.textContent = 'Start Training';
            this.startButton.className = 'btn-primary'; // Blue start button
            this.startButton.disabled = false;
            
            // Enable model selection and epochs input when not training
            this.modelSelect.disabled = false;
            this.epochsInput.disabled = false;
            
            this.hideStatusOverlay();
            if (this.statusText) {
                this.statusText.textContent = 'Ready to start training';
            }
        }
    }

    updateTrainingStatus(status) {
        // Update the training status display
        if (status === "Training") {
            this.fadeOutStatusOverlay();
        } else {
            this.showStatusOverlay(status);
        }
    }

    showStatusOverlay(message) {
        // Show the glass blur overlay with message
        if (this.statusText && this.statusOverlay) {
            this.statusText.textContent = message;
            this.statusOverlay.classList.remove('hidden', 'fade-out');
            this.statusOverlay.classList.add('visible');
        }
    }

    hideStatusOverlay() {
        // Hide the glass blur overlay
        if (this.statusOverlay) {
            this.statusOverlay.classList.remove('visible');
            this.statusOverlay.classList.add('hidden');
        }
    }

    fadeOutStatusOverlay() {
        // Fade out the overlay when training starts
        if (this.statusOverlay) {
            this.statusOverlay.classList.remove('visible');
            this.statusOverlay.classList.add('fade-out');
            setTimeout(() => this.hideStatusOverlay(), 1000);
        }
    }

    simulateTraining(config) {
        // This is a simulation - in real implementation, this would connect to backend
        const totalEpochs = config.epochs;
        let currentEpoch = 0;
        
        const trainingInterval = setInterval(() => {
            currentEpoch++;
            
            // Simulate training metrics
            const progress = (currentEpoch / totalEpochs) * 100;
            const loss = (1.0 - (currentEpoch / totalEpochs)) * Math.random() * 0.5 + 0.1;
            const map = (currentEpoch / totalEpochs) * 0.95 * Math.random() + 0.05;
            
            this.updateMetrics({
                epoch: currentEpoch,
                totalEpochs: totalEpochs,
                loss: loss,
                map: map,
                progress: progress
            });
            
            if (currentEpoch >= totalEpochs) {
                clearInterval(trainingInterval);
                this.trainingComplete();
            }
        }, 1000); // Update every second for demo
    }

    updateMetrics(metrics) {
        
        try {
            // Update epoch displays separately
            if (this.currentEpochDisplay && this.totalEpochsDisplay) {
                this.currentEpochDisplay.textContent = metrics.epoch;
                this.totalEpochsDisplay.textContent = metrics.totalEpochs;
            }
            
            // Calculate epoch progress from training output
            let epochProgress = 0;
            if (metrics.batchProgress !== undefined) {
                epochProgress = metrics.batchProgress;
            } else if (metrics.currentBatch && metrics.totalBatches) {
                epochProgress = (metrics.currentBatch / metrics.totalBatches) * 100;
            }
            
            // Calculate total progress: (currentEpoch-1)/totalEpochs + (epochProgress/100)/totalEpochs
            const totalProgress = ((metrics.epoch - 1) / metrics.totalEpochs + (epochProgress / 100) / metrics.totalEpochs) * 100;
            
            // Update epoch progress bar
            if (this.epochProgressFill && this.epochProgressPercentage) {
                this.epochProgressFill.style.width = `${epochProgress}%`;
                this.epochProgressPercentage.textContent = `${Math.round(epochProgress)}%`;
            }
            
            // Update total progress bar
            if (this.totalProgressFill && this.totalProgressPercentage) {
                this.totalProgressFill.style.width = `${totalProgress}%`;
                this.totalProgressPercentage.textContent = `${Math.round(totalProgress)}%`;
            }
            
            
            // Update GPU memory if available
            if (this.gpuMemoryDisplay) {
                if (metrics.gpuMemory) {
                    this.gpuMemoryDisplay.textContent = metrics.gpuMemory;
                    
                    // Calculate GPU usage percentage (divide by 16GB)
                    const gpuUsagePercentage = this.calculateGpuUsagePercentage(metrics.gpuMemory);
                    this.updateGpuMemoryBar(gpuUsagePercentage);
                    
                    // Store the latest GPU memory value
                    this.lastGpuMemory = metrics.gpuMemory;
                } else if (this.lastGpuMemory) {
                    // Keep showing the last known GPU memory value
                    this.gpuMemoryDisplay.textContent = this.lastGpuMemory;
                    const gpuUsagePercentage = this.calculateGpuUsagePercentage(this.lastGpuMemory);
                    this.updateGpuMemoryBar(gpuUsagePercentage);
                } else {
                    this.gpuMemoryDisplay.textContent = 'N/A';
                    this.updateGpuBar(0);
                }
            }
            
            if (metrics.elapsedTime && metrics.remainingTime) {
                if (this.epochElapsedTime) this.epochElapsedTime.textContent = metrics.elapsedTime;
                if (this.estimatedRemainingTime) this.estimatedRemainingTime.textContent = metrics.remainingTime;
            }
            
        } catch (error) {
            console.error('Error in updateMetrics:', error);
        }
    }
    
    
    
    calculateGpuUsagePercentage(gpuMemoryString) {
        if (!gpuMemoryString || gpuMemoryString === 'N/A' || gpuMemoryString === '-') {
            return 0;
        }
        
        try {
            const match = gpuMemoryString.match(/([0-9.]+)([a-zA-Z])/);
            if (!match) return 0;
            
            const value = parseFloat(match[1]);
            const unit = match[2].toUpperCase();
            
            let valueInGB = value;
            if (unit === 'M' || unit === 'MB') {
                valueInGB = value / 1024;
            } else if (unit === 'K' || unit === 'KB') {
                valueInGB = value / (1024 * 1024);
            }
            
            const percentage = (valueInGB / 16) * 100;
            return Math.min(Math.max(percentage, 0), 100);
            
        } catch (error) {
            return 0;
        }
    }
    
    updateGpuMemoryBar(percentage) {
        if (this.gpuProgressFill && this.gpuPercentageDisplay) {
            this.updateGpuBar(percentage);
            this.gpuPercentageDisplay.textContent = `${Math.round(percentage)}%`;
        }
    }
    
    updateGpuBar(percentage) {
        if (!this.gpuProgressFill) return;
        
        this.gpuProgressFill.style.width = `${percentage}%`;
    }
    
    formatTimeAdvanced(seconds) {
        if (seconds < 0) return '-';
        
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        let timeString = '';
        
        if (days > 0) {
            timeString += `${days.toString().padStart(2, '0')}:`;
        }
        
        if (hours > 0 || days > 0) {
            timeString += `${hours.toString().padStart(2, '0')}:`;
        }
        
        timeString += `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        
        return timeString;
    }
    
   
    parseTimeToSeconds(timeString) {
        if (!timeString || timeString === '-') return 0;
        
        const parts = timeString.split(':');
        if (parts.length === 2) {
            const minutes = parseInt(parts[0]) || 0;
            const seconds = parseInt(parts[1]) || 0;
            return minutes * 60 + seconds;
        }
        return 0;
    }
    
    startTotalElapsedTimer() {
        if (this.totalElapsedTimer) {
            clearInterval(this.totalElapsedTimer);
        }
        this.totalElapsedTimer = setInterval(() => {
            if (this.isTraining) {
                // Update countdown for total remaining time if we have an estimate
                if (this.estimatedTotalRemaining > 0) {
                    this.estimatedTotalRemaining -= 1;
                    this.totalRemainingTime.textContent = this.formatTimeAdvanced(Math.max(0, this.estimatedTotalRemaining));  
                }
            }
        }, 1000);
    }
    
    stopTotalElapsedTimer() {
        if (this.totalElapsedTimer) {
            clearInterval(this.totalElapsedTimer);
            this.totalElapsedTimer = null;
        }
    }
    
    

    trainingComplete() {
        this.isTraining = false;
        this.stopTotalElapsedTimer();
        this.updateUI();
        
        this.epochProgressFill.style.width = '100%';
        this.epochProgressPercentage.textContent = '100%';
        this.totalProgressFill.style.width = '100%';
        this.totalProgressPercentage.textContent = '100%';
        this.epochElapsedTime.textContent = 'Completed';
        this.estimatedRemainingTime.textContent = '00:00';
        this.totalRemainingTime.textContent = '00:00';
    }



    toggleSidebar() {
        this.sidebar.classList.toggle('collapsed');
    }
    
    switchTab(tabName) {
        // Remove active class from all nav items
        this.navItems.forEach(item => item.classList.remove('active'));
        this.tabContents.forEach(content => content.classList.remove('active'));
        
        // Add active class to selected nav item and content
        const selectedNavItem = document.querySelector(`.nav-item[data-tab="${tabName}"]`);
        const selectedContent = document.getElementById(`${tabName}-tab`);
        
        if (selectedNavItem && selectedContent) {
            selectedNavItem.classList.add('active');
            selectedContent.classList.add('active');
        }
        
        // On mobile, close sidebar after selection
        if (window.innerWidth <= 768) {
            this.sidebar.classList.remove('open');
        }
    }
    
    handleMobileResponsive() {
        // Handle mobile sidebar behavior
        if (window.innerWidth <= 768) {
            // On mobile, toggle between open/closed instead of collapsed
            this.sidebarToggle.addEventListener('click', (e) => {
                e.stopPropagation();
                this.sidebar.classList.toggle('open');
            });
            
            // Close sidebar when clicking outside on mobile
            document.addEventListener('click', (e) => {
                if (window.innerWidth <= 768 && 
                    !this.sidebar.contains(e.target) && 
                    this.sidebar.classList.contains('open')) {
                    this.sidebar.classList.remove('open');
                }
            });
        }
    }

    async browseFolder(type) {
        try {
            const response = await fetch('/browse_folder', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ type: type })
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.folder_path) {
                    if (type === 'test') {
                        this.testFilesFolder.value = data.folder_path;
                    } else if (type === 'result') {
                        this.resultFilesFolder.value = data.folder_path;
                    }
                }
            } else {
                console.error('Failed to browse folder');
            }
        } catch (error) {
            console.error('Error browsing folder:', error);
        }
    }

    startDetection() {
        if (this.isTesting) return;

        const testFilesFolder = this.testFilesFolder.value.trim();
        const resultFilesFolder = this.resultFilesFolder.value.trim();
        const selectedModel = this.trainedModelSelect.value;

        if (!this.validateDetectionConfig(testFilesFolder, resultFilesFolder)) return;
        
        if (!selectedModel) {
            alert('Please select a trained model');
            return;
        }

        this.isTesting = true;
        this.updateDetectionUI();
        this.startRealDetection(testFilesFolder, resultFilesFolder, selectedModel);
    }

    validateDetectionConfig(testFilesFolder, resultFilesFolder) {
        if (!testFilesFolder) {
            alert('Please select a test files folder');
            return false;
        }
        if (!resultFilesFolder) {
            alert('Please specify a result files folder');
            return false;
        }
        return true;
    }

    updateDetectionUI() {
        const startButton = this.startDetectionButton;
        const statusDisplay = this.detectionStatusDisplay;
        
        if (this.isTesting) {
            // Update button
            startButton.disabled = true;
            startButton.innerHTML = '<span class="btn-icon">⏳</span><span class="btn-text">Detecting...</span>';
            
            // Update status
            statusDisplay.innerHTML = '<span class="status-indicator">🔄</span><span class="status-text">Running fire detection...</span>';
            statusDisplay.parentElement.style.borderLeft = '4px solid #f6ad55';
        } else {
            // Reset button
            startButton.disabled = false;
            startButton.innerHTML = '<span class="btn-icon">🔥</span><span class="btn-text">Start Detection</span>';
            
            // Reset status
            statusDisplay.innerHTML = '<span class="status-indicator">🟢</span><span class="status-text">Ready to start detection</span>';
            statusDisplay.parentElement.style.borderLeft = '4px solid #48bb78';
        }
    }

    async startRealDetection(testFilesFolder, resultFilesFolder, selectedModel) {
        try {
            // Clear results and show starting message
            this.detectionResultsDisplay.innerHTML = 'Starting detection process...\n';
            
            const response = await fetch('/run_detection', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_name: selectedModel,
                    test_folder: testFilesFolder,
                    output_folder: resultFilesFolder
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Detection request failed');
            }
            
            const result = await response.json();
            this.detectionResultsDisplay.innerHTML += `✓ Detection request sent successfully\n`;
            this.detectionResultsDisplay.innerHTML += `✓ Model: ${result.model_path}\n`;
            this.detectionResultsDisplay.innerHTML += `✓ Source: ${result.test_folder}\n`;
            this.detectionResultsDisplay.innerHTML += `✓ Output: ${result.output_folder}\n\n`;
            
        } catch (error) {
            this.detectionError(error.message);
        }
    }
    
    detectionError(errorMessage) {
        this.isTesting = false;
        this.updateDetectionUI();
        
        const statusDisplay = this.detectionStatusDisplay;
        statusDisplay.innerHTML = '<span class="status-indicator">❌</span><span class="status-text">Detection Failed</span>';
        statusDisplay.parentElement.style.borderLeft = '4px solid #e53e3e';
        
        this.detectionResultsDisplay.innerHTML += `\n❌ Error: ${errorMessage}\n`;
        
        // Reset status after 5 seconds
        setTimeout(() => {
            statusDisplay.innerHTML = '<span class="status-indicator">🟢</span><span class="status-text">Ready to start detection</span>';
            statusDisplay.parentElement.style.borderLeft = '4px solid #48bb78';
        }, 5000);
    }

    detectionComplete(testFilesFolder, resultFilesFolder, selectedModel) {
        this.isTesting = false;
        
        // Update status to completed
        const statusDisplay = this.detectionStatusDisplay;
        statusDisplay.innerHTML = '<span class="status-indicator">✅</span><span class="status-text">Detection Complete!</span>';
        statusDisplay.parentElement.style.borderLeft = '4px solid #48bb78';
        statusDisplay.parentElement.style.background = 'linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%)';
        
        const results = `Detection completed successfully!

Test Files Folder: ${testFilesFolder}
Result Files Folder: ${resultFilesFolder}
Used Model: ${selectedModel}

Detection Summary:
- Total files processed: 15
- Fire detections: 23
- Smoke detections: 45
- Average confidence: 0.842
- Processing time: 5.2 seconds

Results saved to: ${resultFilesFolder}`;

        // Replace the no-results content with actual results
        this.detectionResultsDisplay.innerHTML = results;
        
        // Update the button
        this.updateDetectionUI();
        
        // Reset status after 5 seconds
        setTimeout(() => {
            statusDisplay.innerHTML = '<span class="status-indicator">🟢</span><span class="status-text">Ready to start detection</span>';
            statusDisplay.parentElement.style.background = '';
            statusDisplay.parentElement.style.borderLeft = '4px solid #48bb78';
        }, 5000);
    }

    initializeWebSocket() {
        if (typeof io !== 'undefined') {
            this.socket = io('http://localhost:5000', {
                transports: ['websocket', 'polling']
            });

            this.socket.on('connect', () => {
                console.log('[DEBUG] Socket connected successfully');
            });

            this.socket.on('disconnect', () => {
                console.log('[DEBUG] Socket disconnected');
            });
            
            this.socket.on('training_update', (data) => {
                console.log('[DEBUG] Received training_update:', data);

                this.hideStatusOverlay();
                
                if (!this.isTraining) {
                    this.isTraining = true;
                    this.updateUI();
                }
                
                if (!this.trainingInitialized) {
                    this.trainingInitialized = true;
                }
                
                
                // Calculate total remaining time only once when we don't have an estimate yet
                if (!this.estimatedTotalRemaining && data.elapsed_time && data.remaining_time && data.epoch && data.total_epochs) {
                    const currentEpochElapsed = this.parseTimeToSeconds(data.elapsed_time);
                    const currentEpochRemaining = this.parseTimeToSeconds(data.remaining_time);
                    const currentEpochTotal = currentEpochElapsed + currentEpochRemaining;
                    const remainingEpochs = data.total_epochs - data.epoch;
                    const calculatedTotalRemaining = (remainingEpochs * currentEpochTotal) + currentEpochRemaining;
                    this.estimatedTotalRemaining = calculatedTotalRemaining;
               
                }
                
                this.currentEpoch = data.epoch;
                
                this.updateMetrics({
                    epoch: data.epoch,
                    totalEpochs: data.total_epochs,
                    progress: data.progress,
                    gpuMemory: data.gpu_memory,
                    batchProgress: data.batch_progress,
                    currentBatch: data.current_batch,
                    totalBatches: data.total_batches,
                    iterationSpeed: data.iteration_speed,
                    elapsedTime: data.elapsed_time,
                    remainingTime: data.remaining_time
                });
                
            });
            
            this.socket.on('training_status_update', (data) => {
                console.log('[DEBUG] Received training_status_update:', data);
                this.updateTrainingStatus(data.status);
            });
            
            // Listen for training start
            
            this.socket.on('total_elapsed_update', (data) => {
                if (this.totalElapsedTime && data.total_elapsed_seconds) {
                    this.totalElapsedTime.textContent = this.formatTimeAdvanced(data.total_elapsed_seconds);
                }
            });
            
            // Listen for training completion
            this.socket.on('training_complete', (data) => {
                this.trainingComplete();
            });
            
            this.socket.on('training_error', (data) => {
                this.trainingError(data.error);
            });
            
            // Listen for detection events
            this.socket.on('detection_update', (data) => {
                if (this.isTesting) {
                    this.detectionResultsDisplay.innerHTML += `${data.message}\n`;
                    // Auto-scroll to bottom
                    this.detectionResultsDisplay.scrollTop = this.detectionResultsDisplay.scrollHeight;
                }
            });
            
            this.socket.on('detection_complete', (data) => {
                this.isTesting = false;
                this.updateDetectionUI();
                
                const statusDisplay = this.detectionStatusDisplay;
                statusDisplay.innerHTML = '<span class="status-indicator">✅</span><span class="status-text">Detection Complete!</span>';
                statusDisplay.parentElement.style.borderLeft = '4px solid #48bb78';
                statusDisplay.parentElement.style.background = 'linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%)';
                
                this.detectionResultsDisplay.innerHTML += `\n✅ ${data.message}\n`;
                this.detectionResultsDisplay.innerHTML += `📁 Results saved to: ${data.output_folder}\n`;
                
                // Reset status after 5 seconds
                setTimeout(() => {
                    statusDisplay.innerHTML = '<span class="status-indicator">🟢</span><span class="status-text">Ready to start detection</span>';
                    statusDisplay.parentElement.style.background = '';
                    statusDisplay.parentElement.style.borderLeft = '4px solid #48bb78';
                }, 5000);
            });
            
            this.socket.on('detection_error', (data) => {
                this.detectionError(data.error);
            });
            
        }
    }

    async startRealTraining(config) {
        
        if (!this.socket) {
            this.simulateTraining(config);
            return;
        }

        try {
            const response = await fetch('/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Training request failed');
            }

            const result = await response.json();

        } catch (error) {
            alert(`Training failed: ${error.message}\n\nMake sure the backend server is running on port 5000.`);
            this.trainingError(error.message);
        }
    }

    trainingError(errorMessage) {
        this.isTraining = false;
        this.stopTotalElapsedTimer();
        this.updateUI();
        
        this.epochElapsedTime.textContent = 'Error';
        this.estimatedRemainingTime.textContent = '-';
        this.totalElapsedTime.textContent = 'Error';
        this.totalRemainingTime.textContent = '-';
    }

    simulateTraining(config) {
        const totalEpochs = config.epochs;
        let currentEpoch = 0;
        
        const trainingInterval = setInterval(() => {
                const epochDurationRange = Math.random() * 5000 + 3000; // 3-8 seconds
            
            currentEpoch++;
            
            if (currentEpoch <= totalEpochs) {
                
                const progress = (currentEpoch / totalEpochs) * 100;
                const loss = (1.0 - (currentEpoch / totalEpochs)) * Math.random() * 0.5 + 0.1;
                const map = (currentEpoch / totalEpochs) * 0.95 * Math.random() + 0.05;
                
                this.updateMetrics({
                    epoch: currentEpoch,
                    totalEpochs: totalEpochs,
                    loss: loss,
                    map: map,
                    progress: progress
                });
                
                if (currentEpoch >= totalEpochs) {
                    clearInterval(trainingInterval);
                    this.trainingComplete();
                }
            }
        }, 500); // Check every 500ms for smooth updates
    }

    // Resume Training Dialog Methods
    async checkForUnfinishedTraining() {
        try {
            const response = await fetch('/check_unfinished');
            const data = await response.json();
            
            if (data.has_unfinished && !this.isTraining) {
                this.unfinishedTrainingInfo = data;
                this.showResumeDialog(data);
            }
        } catch (error) {
            console.error('Error checking for unfinished training:', error);
        }
    }

    showResumeDialog(trainingInfo) {
        if (!this.resumeDialog) return;

        // Populate dialog with training information
        if (this.resumeCurrentEpoch) {
            this.resumeCurrentEpoch.textContent = trainingInfo.current_epoch || 0;
        }
        if (this.resumeTotalEpochs) {
            this.resumeTotalEpochs.textContent = trainingInfo.total_epochs || 100;
        }
        if (this.resumeCheckpointType) {
            this.resumeCheckpointType.textContent = trainingInfo.checkpoint_type || 'checkpoint';
        }

        // Update progress bar
        const percentage = trainingInfo.completion_percentage || 0;
        if (this.resumeProgressFill) {
            this.resumeProgressFill.style.width = `${percentage}%`;
        }
        if (this.resumeProgressText) {
            this.resumeProgressText.textContent = `${Math.round(percentage)}%`;
        }

        // Show dialog
        this.resumeDialog.classList.remove('hidden');
    }

    hideResumeDialog() {
        if (this.resumeDialog) {
            this.resumeDialog.classList.add('hidden');
        }
    }

    async resumeTraining() {
        if (!this.unfinishedTrainingInfo) return;

        const additionalEpochs = parseInt(this.resumeAdditionalEpochs?.value) || 50;

        try {
            this.hideResumeDialog();
            
            const response = await fetch('/resume', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    epochs: additionalEpochs
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Resume request failed');
            }

            const result = await response.json();
            
            // Update UI for resumed training
            this.isTraining = true;
            this.updateUI();
            this.initializeTrainingTimers();

        } catch (error) {
            alert(`Failed to resume training: ${error.message}`);
            this.showResumeDialog(this.unfinishedTrainingInfo); // Show dialog again
        }
    }

    async startNewTraining() {
        try {
            this.hideResumeDialog();
            
            // Clear any existing training data
            const clearResponse = await fetch('/clear_training', {
                method: 'POST'
            });

            // Proceed with normal training workflow
            // The user can now use the regular start training button
            
        } catch (error) {
            console.error('Error clearing training data:', error);
            // Continue anyway - user can still start new training
        }
    }

    // Load all models from three sources for training dropdown
    loadAllModels() {
        fetch('/get_all_models')
            .then(response => response.json())
            .then(data => {
                this.modelSelect.innerHTML = ''; // Clear current options
                
                // Add base models section
                if (data.baseModels && data.baseModels.length > 0) {
                    const baseGroup = document.createElement('optgroup');
                    baseGroup.label = 'Fresh Untrained Models';
                    baseGroup.style.color = '#4A90E2'; // Blue
                    data.baseModels.forEach(modelName => {
                        const option = document.createElement('option');
                        option.value = `base:${modelName}`;
                        // Add emoji and clean name (remove Fresh- prefix and .pt extension)
                        let cleanName = modelName.replace(/^Fresh-/i, '').replace(/\.pt$/i, '');
                        option.textContent = `🆕 ${cleanName}`;
                        option.style.color = '#4A90E2'; // Blue
                        baseGroup.appendChild(option);
                    });
                    this.modelSelect.appendChild(baseGroup);
                }
                
                // Add trained models section (includes finished training from Trains folder)
                if (data.trainedModels && data.trainedModels.length > 0) {
                    const trainedGroup = document.createElement('optgroup');
                    trainedGroup.label = 'Completed Training Models';
                    trainedGroup.style.color = '#48bb78'; // Green
                    data.trainedModels.forEach(modelName => {
                        const option = document.createElement('option');
                        option.value = `trained:${modelName}`;
                        // Add emoji and extract timestamp for finished models
                        let displayName;
                        if (modelName.startsWith('finished_run_')) {
                            const timestamp = modelName.replace('finished_run_', '');
                            displayName = `✅ ${timestamp}`;
                        } else {
                            displayName = `✅ ${modelName}`;
                        }
                        option.textContent = displayName;
                        option.style.color = '#48bb78'; // Green
                        trainedGroup.appendChild(option);
                    });
                    this.modelSelect.appendChild(trainedGroup);
                }
                
                // Add checkpoint models section (ongoing training from Trains folder)
                if (data.checkpointModels && data.checkpointModels.length > 0) {
                    const checkpointGroup = document.createElement('optgroup');
                    checkpointGroup.label = 'Ongoing Training (Checkpoints)';
                    checkpointGroup.style.color = '#f6ad55'; // Yellow/Orange
                    data.checkpointModels.forEach(modelName => {
                        const option = document.createElement('option');
                        option.value = `checkpoint:${modelName}`;
                        // Add emoji and extract timestamp for ongoing models
                        let displayName;
                        if (modelName.startsWith('ongoing_run_')) {
                            const timestamp = modelName.replace('ongoing_run_', '');
                            displayName = `🔄 ${timestamp}`;
                        } else {
                            displayName = `🔄 ${modelName}`;
                        }
                        option.textContent = displayName;
                        option.style.color = '#f6ad55'; // Yellow/Orange
                        checkpointGroup.appendChild(option);
                    });
                    this.modelSelect.appendChild(checkpointGroup);
                }
            })
            .catch(error => {
                console.error('Error loading all models:', error);
                this.modelSelect.innerHTML = '<option value="">Error loading models</option>';
            });
    }

    // New function: Load trained models from folder "TrainedModels" via a backend API endpoint
loadTrainedModels() {
    // Fetch the list of model folders (each as a model name)
    fetch('/get_models')
      .then(response => response.json())
      .then(models => {
          this.trainedModelSelect.innerHTML = ''; // Clear current options
          models.forEach(modelName => {
              const option = document.createElement('option');
              option.value = modelName;
              option.textContent = modelName;
              this.trainedModelSelect.appendChild(option);
          });
      })
      .catch(error => {
          console.error('Error loading trained models:', error);
      });
}

loadAppState() {
    fetch('/get_app_state')
        .then(response => response.json())
        .then(state => {
            // Restore debug logging checkbox state
            if (this.debugLoggingCheckbox) {
                this.debugLoggingCheckbox.checked = state.debug_logging || false;
            }
            
            // Restore training state
            if (state.training_status !== 'idle') {
                console.log('Restoring training state:', state.training_status);
                
                // Update training UI based on state
                this.currentEpochDisplay.textContent = state.current_epoch || '-';
                this.totalEpochsDisplay.textContent = state.total_epochs || '-';
                this.statusText.textContent = state.last_training_message || 'Training...';
                
                // If training is active (not idle or failed), set UI to training mode
                if (state.training_status !== 'idle' && state.training_status !== 'failed') {
                    this.isTraining = true;
                    this.updateUI(); // Button becomes "Cancel Training"
                    this.initializeTrainingTimers(); // Start the timer for total remaining time
                }
                // If idle or failed, keep default UI (button stays "Start Training")
            }
        })
        .catch(error => {
            console.error('Error loading app state:', error);
        });
}

// New function: Rearrange detection controls into two rows
reorderDetectionUI() {
    // Create container rows
    const row1 = document.createElement('div');
    row1.classList.add('detection-row');
    const row2 = document.createElement('div');
    row2.classList.add('detection-row');

    // Assume these elements have already been captured in initializeElements:
    // this.testFilePathInput, this.outputResultsPathInput, this.trainedModelSelect, this.startDetectionButton
    const testFolderContainer = this.testFilePathInput.parentNode;
    const outputFolderContainer = this.outputResultsPathInput.parentNode;
    const trainedModelContainer = this.trainedModelSelect.parentNode;
    const startDetectionContainer = this.startDetectionButton.parentNode;

    // Append test folder and output folder to the first row
    row1.appendChild(testFolderContainer);
    row1.appendChild(outputFolderContainer);

    // Append trained model select and start detection button to the second row
    row2.appendChild(trainedModelContainer);
    row2.appendChild(startDetectionContainer);

    // Assuming there's a container with id 'detection-container' that holds these controls
    const detectionContainer = document.getElementById('detection-container');
    if (detectionContainer) {
        detectionContainer.innerHTML = ''; // Clear current layout
        detectionContainer.appendChild(row1);
        detectionContainer.appendChild(row2);
    }
}
    // ===============================================
    // LIVE DETECTION FUNCTIONALITY
    // ===============================================
    
    initializeLiveDetection() {
        this.isLiveDetectionActive = false;
        this.currentStream = null;
        this.detectionCounts = { fire: 0, smoke: 0, other: 0 };
        this.detectionInterval = null;
        this.loadAvailableCameras();
        this.loadModelsForLiveDetection();
        this.setupLiveDetectionEvents();
    }
    
    async loadAvailableCameras() {
        try {
            // Check if mediaDevices is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
                this.cameraSelect.innerHTML = '<option value="">Camera access not supported</option>';
                this.showCameraError('Media devices not supported in this browser');
                return;
            }
            
            console.log('Loading available cameras...');
            
            // Try to enumerate devices without requesting permission first
            // The backend will handle camera access when detection starts
            let devices, videoDevices;
            
            try {
                devices = await navigator.mediaDevices.enumerateDevices();
                videoDevices = devices.filter(device => device.kind === 'videoinput');
                
                // If no devices found or labels are empty, try requesting permission
                if (videoDevices.length === 0 || videoDevices.every(d => !d.label)) {
                    console.log('No labeled cameras found, requesting permission...');
                    
                    // Request permission to get device labels
                    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    stream.getTracks().forEach(track => track.stop());
                    
                    // Re-enumerate with labels
                    devices = await navigator.mediaDevices.enumerateDevices();
                    videoDevices = devices.filter(device => device.kind === 'videoinput');
                }
            } catch (permissionError) {
                console.log('Camera enumeration failed:', permissionError);
                // Still show basic camera options even without permission
                videoDevices = [{ deviceId: '0', label: 'Default Camera' }];
            }
            
            // Clear existing options
            this.cameraSelect.innerHTML = '';
            
            if (videoDevices.length === 0) {
                this.cameraSelect.innerHTML = '<option value="">No cameras detected</option>';
                this.showCameraError('No cameras found on this device');
                return;
            }
            
            // Add default selection option
            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = 'Select a camera...';
            this.cameraSelect.appendChild(defaultOption);
            
            // Add camera options
            videoDevices.forEach((device, index) => {
                const option = document.createElement('option');
                option.value = index;  // Use index for backend compatibility
                option.textContent = device.label || `Camera ${index + 1}`;
                this.cameraSelect.appendChild(option);
            });
            
            console.log(`Found ${videoDevices.length} camera(s)`);
            
        } catch (error) {
            console.error('Error loading cameras:', error);
            this.cameraSelect.innerHTML = '<option value="">Error loading cameras</option>';
            this.showCameraError('Failed to access camera devices');
        }
    }
    
    showCameraAccessError(errorMessage) {
        // Create camera error UI with specific error information
        const errorDiv = document.createElement('div');
        errorDiv.className = 'camera-access-error';
        
        let errorContent = '';
        let solutions = [];
        
        if (errorMessage.includes('Permission denied') || errorMessage.includes('NotAllowedError')) {
            errorContent = '🚫 Camera Permission Denied';
            solutions = [
                'Click the camera icon in your browser\'s address bar',
                'Select "Allow" for camera access',
                'Refresh the page'
            ];
        } else if (errorMessage.includes('NotFoundError') || errorMessage.includes('DevicesNotFoundError')) {
            errorContent = '📷 No Camera Found';
            solutions = [
                'Make sure your camera is connected',
                'Check if other applications are using the camera',
                'Try refreshing the page'
            ];
        } else if (errorMessage.includes('OverconstrainedError') || errorMessage.includes('constraints could not be satisfied')) {
            errorContent = '⚠️ Camera Constraints Error';
            solutions = [
                'Your camera might not support the requested resolution',
                'Try closing other camera applications',
                'Make sure your camera drivers are up to date',
                'Try refreshing the page'
            ];
        } else {
            errorContent = '❌ Camera Access Error';
            solutions = [
                'Make sure your camera is connected and working',
                'Close other applications that might be using the camera',
                'Try refreshing the page'
            ];
        }
        
        errorDiv.innerHTML = `
            <div style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); 
                        border: 2px solid #f44336; 
                        border-radius: 12px; 
                        padding: 20px; 
                        margin: 15px 0; 
                        color: #c62828;
                        box-shadow: 0 4px 12px rgba(244, 67, 54, 0.2);">
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 24px; margin-right: 12px;">📷</span>
                    <strong style="font-size: 18px;">${errorContent}</strong>
                </div>
                <p style="margin: 10px 0; line-height: 1.6; background: rgba(255,255,255,0.7); padding: 8px; border-radius: 6px; font-family: monospace; font-size: 12px;">
                    Error: ${errorMessage}
                </p>
                <p style="margin: 10px 0; line-height: 1.6;">
                    Please try the following solutions:
                </p>
                <ol style="margin: 10px 0; padding-left: 20px; line-height: 1.6;">
                    ${solutions.map(solution => `<li>${solution}</li>`).join('')}
                </ol>
                <div style="display: flex; gap: 10px; margin-top: 15px;">
                    <button onclick="location.reload()" 
                            style="background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); 
                                   color: white; 
                                   border: none; 
                                   padding: 12px 24px; 
                                   border-radius: 8px; 
                                   cursor: pointer; 
                                   font-weight: bold;
                                   box-shadow: 0 3px 8px rgba(0, 123, 255, 0.3);
                                   transition: all 0.3s ease;"
                            onmouseover="this.style.transform='translateY(-2px)'"
                            onmouseout="this.style.transform='translateY(0)'">
                        🔄 Retry
                    </button>
                    <button onclick="window.cameraTestFunction()" 
                            style="background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%); 
                                   color: white; 
                                   border: none; 
                                   padding: 12px 24px; 
                                   border-radius: 8px; 
                                   cursor: pointer; 
                                   font-weight: bold;
                                   box-shadow: 0 3px 8px rgba(40, 167, 69, 0.3);
                                   transition: all 0.3s ease;"
                            onmouseover="this.style.transform='translateY(-2px)'"
                            onmouseout="this.style.transform='translateY(0)'">
                        🧪 Test Camera
                    </button>
                </div>
            </div>
        `;
        
        // Insert after camera select card
        const cameraCard = this.cameraSelect.closest('.camera-select-card');
        if (cameraCard && cameraCard.parentNode) {
            cameraCard.parentNode.insertBefore(errorDiv, cameraCard.nextSibling);
        }
        
        // Add global camera test function
        window.cameraTestFunction = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                alert('✅ Camera test successful! Your camera is working. Try reloading the page.');
                stream.getTracks().forEach(track => track.stop());
            } catch (error) {
                alert(`❌ Camera test failed: ${error.message}`);
            }
        };
    }
    
    loadModelsForLiveDetection() {
        // Load models for live detection (reuse the same endpoint)
        fetch('/get_models')
            .then(response => response.json())
            .then(models => {
                this.liveModelSelect.innerHTML = '';
                if (models.length === 0) {
                    this.liveModelSelect.innerHTML = '<option value="">No trained models found</option>';
                } else {
                    models.forEach(modelName => {
                        const option = document.createElement('option');
                        option.value = modelName;
                        option.textContent = modelName;
                        this.liveModelSelect.appendChild(option);
                    });
                }
            })
            .catch(error => {
                console.error('Error loading models for live detection:', error);
                this.liveModelSelect.innerHTML = '<option value="">Error loading models</option>';
            });
    }
    
    async toggleLiveDetection() {
        if (this.isLiveDetectionActive) {
            this.stopLiveDetection();
        } else {
            await this.startLiveDetection();
        }
    }
    
    async startLiveDetection() {
        try {
            const selectedCamera = this.cameraSelect.value;
            const selectedModel = this.liveModelSelect.value;
            
            if (!selectedCamera) {
                alert('Please select a camera first');
                return;
            }
            
            if (!selectedModel) {
                alert('Please select a detection model first');
                return;
            }
            
            console.log('Starting live detection with:', { selectedCamera, selectedModel });
            
            // Don't start local camera - let backend handle it
            // Just start backend detection which will handle camera access
            const response = await fetch('/start_live_detection', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_name: selectedModel,
                    camera_index: parseInt(selectedCamera) || 0
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Failed to start live detection');
            }
            
            const result = await response.json();
            console.log('Backend detection started:', result);
            
            // Hide placeholder and show that detection is starting
            this.cameraPlaceholder.style.display = 'none';
            this.cameraVideo.style.display = 'none';
            this.processedFrame.style.display = 'block';
            
            // Show loading state
            this.processedFrame.src = 'data:image/svg+xml;base64,' + btoa(`
                <svg width="640" height="480" xmlns="http://www.w3.org/2000/svg">
                    <rect width="100%" height="100%" fill="#2c3e50"/>
                    <text x="50%" y="40%" text-anchor="middle" fill="white" font-size="24" font-family="Arial">
                        🤖 Loading AI Detection...
                    </text>
                    <text x="50%" y="60%" text-anchor="middle" fill="#74b9ff" font-size="16" font-family="Arial">
                        Model: ${selectedModel}
                    </text>
                </svg>
            `);
            
            this.isLiveDetectionActive = true;
            this.updateLiveDetectionUI();
            
        } catch (error) {
            console.error('Error starting live detection:', error);
            this.showCameraError(error.message);
        }
    }
    
    async initializeCamera(deviceId = null) {
        try {
            // Stop any existing stream
            this.stopCamera();
            
            // Request camera permission and stream
            const constraints = {
                video: {
                    deviceId: deviceId ? { exact: deviceId } : undefined,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };
            
            this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Set video source
            this.cameraVideo.srcObject = this.currentStream;
            
            // Wait for video to load
            await new Promise((resolve, reject) => {
                this.cameraVideo.onloadedmetadata = () => {
                    resolve();
                };
                this.cameraVideo.onerror = () => {
                    reject(new Error('Failed to load camera video'));
                };
            });
            
            // Hide placeholder and error, show video
            this.hideCameraError();
            this.cameraPlaceholder.style.display = 'none';
            this.cameraVideo.style.display = 'block';
            
            // Update status
            this.updateFeedStatus(true);
            
            // Setup canvas for detection overlay
            this.setupDetectionCanvas();
            
        } catch (error) {
            console.error('Camera initialization error:', error);
            
            let errorMsg = 'Unable to access camera';
            if (error.name === 'NotAllowedError') {
                errorMsg = 'Camera permission denied. Please allow camera access and try again.';
            } else if (error.name === 'NotFoundError') {
                errorMsg = 'No camera found. Please connect a camera and try again.';
            } else if (error.name === 'NotReadableError') {
                errorMsg = 'Camera is already in use by another application.';
            }
            
            this.showCameraError(errorMsg);
            throw error;
        }
    }
    
    setupDetectionCanvas() {
        const canvas = this.detectionCanvas;
        const video = this.cameraVideo;
        
        // Set canvas size to match video
        canvas.width = video.videoWidth || video.offsetWidth;
        canvas.height = video.videoHeight || video.offsetHeight;
        
        // Position canvas over video
        canvas.style.width = '100%';
        canvas.style.height = '100%';
    }
    
    setupLiveDetectionEvents() {
        // Initialize fire alert system
        this.fireAlertTimeout = null;
        this.fireWarningTimeout = null;
        this.originalPageBackground = document.body.style.background || '';
        
        // Listen for processed frames with detections
        this.socket.on('live_detection_frame', (data) => {
            this.handleLiveDetectionFrame(data);
        });
        
        this.socket.on('live_detection_started', (data) => {
            this.updateFeedStatus(true, data.message);
        });
        
        this.socket.on('live_detection_stopped', (data) => {
            this.updateFeedStatus(false, data.message);
            this.resetPageBackground();
        });
        
        this.socket.on('live_detection_error', (data) => {
            this.showCameraError(data.error);
            this.resetPageBackground();
        });
    }
    
    handleLiveDetectionFrame(data) {
        // Display processed frame with bounding boxes
        if (data.frame) {
            // Hide raw camera feed and show processed frame
            this.cameraVideo.style.display = 'none';
            this.processedFrame.style.display = 'block';
            this.processedFrame.src = 'data:image/jpeg;base64,' + data.frame;
            
            // Hide placeholder
            this.cameraPlaceholder.style.display = 'none';
        }
        
        // Process detection results
        const detections = data.detections;
        const counts = data.counts;
        
        // Update detection counts
        this.detectionCounts = counts;
        this.updateDetectionCountsDisplay();
        
        // Check for fire detection and trigger alerts
        const fireDetections = detections.filter(d => d.class === 'fire');
        if (fireDetections.length > 0) {
            this.triggerFireAlert();
        }
        
        // Process each detection for logging
        detections.forEach(detection => {
            const detectionType = detection.class;
            const confidence = detection.confidence;
            
            // Add to detection log
            this.addDetectionEntry(detectionType, confidence);
        });
    }
    
    triggerFireAlert() {
        // Clear any existing timeouts
        if (this.fireWarningTimeout) {
            clearTimeout(this.fireWarningTimeout);
        }
        if (this.fireAlertTimeout) {
            clearTimeout(this.fireAlertTimeout);
        }
        
        // Immediately set background to yellow (warning)
        document.body.style.background = 'linear-gradient(135deg, #fff3a0 0%, #fed02f 100%)';
        document.body.style.transition = 'background 0.5s ease';
        
        // After 2 seconds, change to red (alert)
        this.fireWarningTimeout = setTimeout(() => {
            document.body.style.background = 'linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)';
            
            // After 5 total seconds, reset to normal
            this.fireAlertTimeout = setTimeout(() => {
                this.resetPageBackground();
            }, 3000); // 3 more seconds (5 total)
        }, 2000);
    }
    
    resetPageBackground() {
        // Clear timeouts
        if (this.fireWarningTimeout) {
            clearTimeout(this.fireWarningTimeout);
            this.fireWarningTimeout = null;
        }
        if (this.fireAlertTimeout) {
            clearTimeout(this.fireAlertTimeout);
            this.fireAlertTimeout = null;
        }
        
        // Reset background to original
        document.body.style.background = this.originalPageBackground;
        document.body.style.transition = 'background 0.5s ease';
    }
    
    addDetectionEntry(type, confidence) {
        const timestamp = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = `detection-entry ${type}`;
        entry.innerHTML = `
            <div class="detection-time">${timestamp}</div>
            <div class="detection-info">${type.toUpperCase()} detected (${(confidence * 100).toFixed(0)}% confidence)</div>
        `;
        
        // Remove no-detections message if it exists
        const noDetections = this.liveDetectionLog.querySelector('.no-detections');
        if (noDetections) {
            noDetections.remove();
        }
        
        // Add new entry at the top
        this.liveDetectionLog.insertBefore(entry, this.liveDetectionLog.firstChild);
        
        // Keep only last 20 entries
        const entries = this.liveDetectionLog.querySelectorAll('.detection-entry');
        if (entries.length > 20) {
            entries[entries.length - 1].remove();
        }
    }
    
    updateDetectionCounts(type) {
        this.detectionCounts[type]++;
        this.fireCountDisplay.textContent = this.detectionCounts.fire;
        this.smokeCountDisplay.textContent = this.detectionCounts.smoke;
    }
    
    drawDetectionBox(type, bbox = null) {
        const canvas = this.detectionCanvas;
        const ctx = canvas.getContext('2d');
        
        let x, y, width, height;
        
        if (bbox) {
            // Use real bounding box coordinates from YOLO
            const [x1, y1, x2, y2] = bbox;
            x = x1;
            y = y1;
            width = x2 - x1;
            height = y2 - y1;
            
            // Scale coordinates to canvas size (assuming 640x480 input)
            const scaleX = canvas.width / 640;
            const scaleY = canvas.height / 480;
            
            x *= scaleX;
            y *= scaleY;
            width *= scaleX;
            height *= scaleY;
        } else {
            // Fallback to random position for demo
            x = Math.random() * (canvas.width - 100);
            y = Math.random() * (canvas.height - 100);
            width = 80 + Math.random() * 40;
            height = 60 + Math.random() * 30;
        }
        
        // Set colors for different detection types
        const colors = {
            fire: '#ff6b6b',
            smoke: '#74b9ff',
            other: '#55a3ff'
        };
        
        const color = colors[type] || '#00ff00';
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);
        
        // Draw label
        ctx.fillStyle = color;
        ctx.font = '14px Arial';
        ctx.fillText(type.toUpperCase(), x, y - 5);
        
        // Clear detection box after 3 seconds
        setTimeout(() => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }, 3000);
    }
    
    stopLiveDetection() {
        this.isLiveDetectionActive = false;
        this.stopCamera();
        this.updateLiveDetectionUI();
        
        // Stop backend detection
        fetch('/stop_live_detection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        }).catch(error => {
            console.error('Error stopping live detection:', error);
        });
        
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
        
        // Show placeholder and hide processed frame
        this.cameraPlaceholder.style.display = 'flex';
        this.cameraVideo.style.display = 'none';
        this.processedFrame.style.display = 'none';
        this.updateFeedStatus(false);
        
        // Clear detection canvas
        const ctx = this.detectionCanvas.getContext('2d');
        ctx.clearRect(0, 0, this.detectionCanvas.width, this.detectionCanvas.height);
    }
    
    stopCamera() {
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => {
                track.stop();
            });
            this.currentStream = null;
        }
    }
    
    updateLiveDetectionUI() {
        const button = this.startLiveDetectionButton;
        
        if (this.isLiveDetectionActive) {
            button.innerHTML = '<span class="btn-icon">⏹️</span><span class="btn-text">Stop Camera</span>';
            button.disabled = false;
        } else {
            button.innerHTML = '<span class="btn-icon">🔴</span><span class="btn-text">Start Camera</span>';
            button.disabled = false;
        }
    }
    
    updateFeedStatus(isOnline) {
        const statusDot = this.feedStatus.querySelector('.status-dot');
        const statusText = this.feedStatus.querySelector('.status-text');
        
        if (isOnline) {
            statusDot.className = 'status-dot online';
            statusText.textContent = 'Camera Online';
        } else {
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'Camera Offline';
        }
    }
    
    showCameraError(message) {
        this.errorMessage.textContent = message;
        this.cameraError.classList.remove('hidden');
        this.cameraPlaceholder.style.display = 'none';
        this.cameraVideo.style.display = 'none';
        this.updateFeedStatus(false);
    }
    
    hideCameraError() {
        this.cameraError.classList.add('hidden');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    try {
        window.dashboard = new TrainingDashboard();
    } catch (error) {
        console.error('Failed to create dashboard:', error);
        alert('Failed to initialize dashboard: ' + error.message);
    }
});