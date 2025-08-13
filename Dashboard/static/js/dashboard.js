class TrainingDashboard {
    constructor() {
        this.isTraining = false;
        this.isTesting = false;
        this.socket = null;
        this.unfinishedTrainingInfo = null;
        
        this.initializeElements();
        this.attachEventListeners();
        this.initializeWebSocket();
        this.loadTrainedModels();
        this.checkForUnfinishedTraining();
    }

    initializeElements() {
        // Tab elements
        this.tabButtons = document.querySelectorAll('.tab-button');
        this.tabContents = document.querySelectorAll('.tab-content');
        
        // Training configuration elements
        this.yoloVersionSelect = document.getElementById('yolo-version');
        this.modelSizeSelect = document.getElementById('model-size');
        this.epochsInput = document.getElementById('epochs');
        this.startButton = document.getElementById('start-training');
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
        
        this.testFilePathInput = document.getElementById('test-file-path');
        this.outputResultsPathInput = document.getElementById('output-results-path');
        this.browseTestFileButton = document.getElementById('browse-test-file');
        this.browseOutputPathButton = document.getElementById('browse-output-path');
        this.startDetectionButton = document.getElementById('start-detection');
        this.detectionStatusDisplay = document.getElementById('detection-status');
        this.detectionResultsDisplay = document.getElementById('detection-results');
        
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
        this.browseTestFolderButton = document.getElementById('browse-test-folder');
        this.browseOutputFolderButton = document.getElementById('browse-output-folder');
        
    }

    attachEventListeners() {
        this.tabButtons.forEach(button => {
            button.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
        
        if (this.startButton) {
            this.startButton.addEventListener('click', () => this.startTraining());
        }
        
        
        this.browseTestFolderButton.addEventListener('click', () => this.browseFolder('test'));
        this.browseOutputFolderButton.addEventListener('click', () => this.browseFolder('output'));
        this.startDetectionButton.addEventListener('click', () => this.startDetection());
        
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
            yolo_version: parseInt(this.yoloVersionSelect.value),
            model_size: this.modelSizeSelect.value,
            epochs: parseInt(this.epochsInput.value)
        };

        if (!this.validateConfig(config)) return;

        this.isTraining = true;
        this.initializeTrainingTimers();
        this.updateUI();
        this.startRealTraining(config);
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
        if (config.epochs < 1 || config.epochs > 1000) {
            alert('Please enter a valid number of epochs (1-1000)');
            return false;
        }
        return true;
    }

    updateUI() {
        this.startButton.disabled = this.isTraining;
        this.startButton.textContent = this.isTraining ? 'Training...' : 'Start Training';
        
        if (this.isTraining) {
            this.showStatusOverlay('Preparing for train : Loading YOLO...');
        } else {
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
                if (this.estimatedTotalRemaining && this.estimatedTotalRemaining > 0) {
                    this.estimatedTotalRemaining -= 1;
                    if (this.totalRemainingTime) {
                        this.totalRemainingTime.textContent = this.formatTimeAdvanced(Math.max(0, this.estimatedTotalRemaining));
                    }
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



    switchTab(tabName) {
        this.tabButtons.forEach(button => button.classList.remove('active'));
        this.tabContents.forEach(content => content.classList.remove('active'));
        
        const selectedButton = document.querySelector(`[data-tab="${tabName}"]`);
        const selectedContent = document.getElementById(`${tabName}-tab`);
        
        if (selectedButton && selectedContent) {
            selectedButton.classList.add('active');
            selectedContent.classList.add('active');
        }
    }

    browseFile(type) {
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        
        if (type === 'test') {
            fileInput.accept = '.jpg,.jpeg,.png,.mp4,.avi,.mov,.wmv';
        } else if (type === 'output') {
            fileInput.accept = '*';
        }
        
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                if (type === 'test') {
                    this.testFilePathInput.value = file.name; // In real app, would be full path
                } else if (type === 'output') {
                    const fileName = file.name;
                    const outputPath = fileName.substring(0, fileName.lastIndexOf('.')) + '_results.mp4';
                    this.outputResultsPathInput.value = outputPath;
                }
            }
        });
        
        fileInput.click();
    }

    startDetection() {
        if (this.isTesting) return;

        const testFilePath = this.testFilePathInput.value.trim();
        const outputResultsPath = this.outputResultsPathInput.value.trim();

        if (!this.validateDetectionConfig(testFilePath, outputResultsPath)) return;

        this.isTesting = true;
        this.updateDetectionUI();
        this.simulateDetection(testFilePath, outputResultsPath);
    }

    validateDetectionConfig(testFilePath, outputResultsPath) {
        if (!testFilePath) {
            alert('Please select a test file');
            return false;
        }
        if (!outputResultsPath) {
            alert('Please specify an output path for results');
            return false;
        }
        return true;
    }

    updateDetectionUI() {
        this.startDetectionButton.disabled = this.isTesting;
        this.startDetectionButton.textContent = this.isTesting ? 'Detecting...' : 'Start Detection';
        
        if (this.isTesting) {
            this.detectionStatusDisplay.textContent = 'Running fire detection...';
            this.detectionStatusDisplay.classList.add('training'); // Reuse training styles for animation
        } else {
            this.detectionStatusDisplay.textContent = 'Ready to start detection';
            this.detectionStatusDisplay.classList.remove('training');
        }
    }

    simulateDetection(testFilePath, outputResultsPath) {
        const detectionSteps = [
            'Loading trained model...',
            'Processing input file...',
            'Analyzing frames for fire and smoke...',
            'Applying detection algorithms...',
            'Generating results...',
            'Saving output file...'
        ];

        let currentStep = 0;
        this.detectionResultsDisplay.textContent = 'Starting detection process...\n';

        const detectionInterval = setInterval(() => {
            if (currentStep < detectionSteps.length) {
                this.detectionResultsDisplay.textContent += `✓ ${detectionSteps[currentStep]}\n`;
                currentStep++;
            } else {
                clearInterval(detectionInterval);
                this.detectionComplete(testFilePath, outputResultsPath);
            }
        }, 1000);
    }

    detectionComplete(testFilePath, outputResultsPath) {
        this.isTesting = false;
        this.updateDetectionUI();
        
        this.detectionStatusDisplay.textContent = 'Detection Complete!';
        this.detectionStatusDisplay.style.background = 'linear-gradient(135deg, #48bb78 0%, #38a169 100%)';
        this.detectionStatusDisplay.style.color = 'white';
        
        const results = `
Detection completed successfully!

Input File: ${testFilePath}
Output File: ${outputResultsPath}

Detection Summary:
- Total frames processed: 1,247
- Fire detections: 23
- Smoke detections: 45
- Average confidence: 0.842
- Processing time: 5.2 seconds

Results saved to: ${outputResultsPath}`;

        this.detectionResultsDisplay.textContent += '\n' + results;
        
        setTimeout(() => {
            this.detectionStatusDisplay.textContent = 'Ready to start detection';
            this.detectionStatusDisplay.style.background = '';
            this.detectionStatusDisplay.style.color = '';
        }, 5000);
    }

    initializeWebSocket() {
        if (typeof io !== 'undefined') {
            this.socket = io('http://localhost:5000', {
                transports: ['websocket', 'polling']
            });
            
            this.socket.on('training_update', (data) => {
                
                this.hideStatusOverlay();
                
                if (!this.isTraining) {
                    this.isTraining = true;
                    this.updateUI();
                }
                
                if (!this.trainingInitialized) {
                    this.trainingInitialized = true;
                }
                
                
                if (data.total_remaining_seconds !== null && data.total_remaining_seconds !== undefined && this.totalRemainingTime) {
                    this.estimatedTotalRemaining = data.total_remaining_seconds;
                    this.totalRemainingTime.textContent = this.formatTimeAdvanced(data.total_remaining_seconds);
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
            
            
            // Listen for connection status
            
            
            
            
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

}

document.addEventListener('DOMContentLoaded', () => {
    try {
        window.dashboard = new TrainingDashboard();
    } catch (error) {
        console.error('Failed to create dashboard:', error);
        alert('Failed to initialize dashboard: ' + error.message);
    }
});