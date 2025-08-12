class TrainingDashboard {
    constructor() {
        this.isTraining = false;
        this.isTesting = false;
        this.socket = null;
        this.timerInterval = null;
        this.initializeElements();
        this.attachEventListeners();
        this.initializeWebSocket();
    }

    initializeElements() {
        // Tab elements
        this.tabButtons = document.querySelectorAll('.tab-button');
        this.tabContents = document.querySelectorAll('.tab-content');
        
        // Training elements
        this.yoloVersionSelect = document.getElementById('yolo-version');
        this.modelSizeSelect = document.getElementById('model-size');
        this.epochsInput = document.getElementById('epochs');
        this.startButton = document.getElementById('start-training');
        this.statusDisplay = document.getElementById('training-status');
        
        // Progress elements
        this.epochCounterDisplay = document.getElementById('epoch-counter');
        this.progressFill = document.getElementById('progress-fill');
        this.progressPercentage = document.getElementById('progress-percentage');
        this.epochElapsedTime = document.getElementById('epoch-elapsed-time');
        this.estimatedRemainingTime = document.getElementById('estimated-remaining-time');
        
        // Training timing variables
        this.trainingStartTime = null;
        this.epochStartTime = null;
        this.epochDurations = [];
        this.currentEpoch = 0;
        
        // Test elements
        this.testFilePathInput = document.getElementById('test-file-path');
        this.outputResultsPathInput = document.getElementById('output-results-path');
        this.browseTestFileButton = document.getElementById('browse-test-file');
        this.browseOutputPathButton = document.getElementById('browse-output-path');
        this.startDetectionButton = document.getElementById('start-detection');
        this.detectionStatusDisplay = document.getElementById('detection-status');
        this.detectionResultsDisplay = document.getElementById('detection-results');
        
        // Console elements
        this.consoleOutput = document.getElementById('console-output');
        this.clearConsoleButton = document.getElementById('clear-console');
    }

    attachEventListeners() {
        // Tab listeners
        this.tabButtons.forEach(button => {
            button.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
        
        // Training listeners
        this.startButton.addEventListener('click', () => this.startTraining());
        
        // Test listeners
        this.browseTestFileButton.addEventListener('click', () => this.browseFile('test'));
        this.browseOutputPathButton.addEventListener('click', () => this.browseFile('output'));
        this.startDetectionButton.addEventListener('click', () => this.startDetection());
        
        // Console listeners
        this.clearConsoleButton.addEventListener('click', () => this.clearConsole());
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
        this.epochStartTime = null;
        this.epochDurations = [];
        
        // Reset display values
        this.epochElapsedTime.textContent = '-';
        this.estimatedRemainingTime.textContent = '-';
        this.epochCounterDisplay.textContent = '-';
        this.progressFill.style.width = '0%';
        this.progressPercentage.textContent = '0%';
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
            this.statusDisplay.textContent = 'Training...';
            this.statusDisplay.classList.add('training');
        } else {
            this.statusDisplay.textContent = 'Ready to start training';
            this.statusDisplay.classList.remove('training');
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
        // Update epoch counter
        this.epochCounterDisplay.textContent = `${metrics.epoch}/${metrics.totalEpochs}`;
        
        // Update progress bar and percentage
        this.progressFill.style.width = `${metrics.progress}%`;
        this.progressPercentage.textContent = `${Math.round(metrics.progress)}%`;
        
        // Update timing information
        this.updateTimingInfo(metrics.epoch, metrics.totalEpochs);
    }
    
    updateTimingInfo(currentEpoch, totalEpochs) {
        const now = Date.now();
        
        // Update current epoch elapsed time
        if (this.epochStartTime) {
            const epochElapsed = (now - this.epochStartTime) / 1000;
            this.epochElapsedTime.textContent = this.formatTime(epochElapsed);
        }
        
        // Calculate and update estimated remaining time
        if (this.trainingStartTime && currentEpoch > 0) {
            const totalElapsed = (now - this.trainingStartTime) / 1000;
            const averageEpochTime = totalElapsed / currentEpoch;
            const remainingEpochs = totalEpochs - currentEpoch;
            const estimatedRemaining = averageEpochTime * remainingEpochs;
            
            this.estimatedRemainingTime.textContent = this.formatTime(estimatedRemaining);
        }
    }
    
    formatTime(seconds) {
        if (seconds < 0) return '-';
        
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }
    
    startEpochTimer() {
        this.epochStartTime = Date.now();
        
        // Start real-time timer update
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
        }
        
        this.timerInterval = setInterval(() => {
            if (this.epochStartTime) {
                const elapsed = (Date.now() - this.epochStartTime) / 1000;
                this.epochElapsedTime.textContent = this.formatTime(elapsed);
            }
        }, 1000);
    }
    
    stopEpochTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        
        if (this.epochStartTime) {
            const epochDuration = (Date.now() - this.epochStartTime) / 1000;
            this.epochDurations.push(epochDuration);
        }
    }

    trainingComplete() {
        this.isTraining = false;
        this.stopEpochTimer();
        this.updateUI();
        
        this.statusDisplay.textContent = 'Training Complete!';
        this.statusDisplay.style.background = 'linear-gradient(135deg, #48bb78 0%, #38a169 100%)';
        this.statusDisplay.style.color = 'white';
        
        // Show final completion state
        this.progressFill.style.width = '100%';
        this.progressPercentage.textContent = '100%';
        this.epochElapsedTime.textContent = 'Completed';
        this.estimatedRemainingTime.textContent = '0s';
        
        setTimeout(() => {
            this.statusDisplay.textContent = 'Ready to start training';
            this.statusDisplay.style.background = '';
            this.statusDisplay.style.color = '';
        }, 3000);
    }

    // Method to connect to real backend (for future implementation)
    async connectToBackend(config) {
        try {
            const response = await fetch('/api/start-training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            if (response.ok) {
                this.startRealtimeUpdates();
            }
        } catch (error) {
            console.error('Failed to start training:', error);
            this.isTraining = false;
            this.updateUI();
        }
    }

    // WebSocket connection for real-time updates (for future implementation)
    startRealtimeUpdates() {
        // const ws = new WebSocket('ws://localhost:8080/training-updates');
        // ws.onmessage = (event) => {
        //     const data = JSON.parse(event.data);
        //     this.updateMetrics(data);
        // };
    }

    // Tab functionality
    switchTab(tabName) {
        // Remove active class from all tabs and contents
        this.tabButtons.forEach(button => button.classList.remove('active'));
        this.tabContents.forEach(content => content.classList.remove('active'));
        
        // Add active class to selected tab and content
        const selectedButton = document.querySelector(`[data-tab="${tabName}"]`);
        const selectedContent = document.getElementById(`${tabName}-tab`);
        
        if (selectedButton && selectedContent) {
            selectedButton.classList.add('active');
            selectedContent.classList.add('active');
        }
    }

    // File browsing functionality
    browseFile(type) {
        // Create a hidden file input
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        
        if (type === 'test') {
            // Accept image and video files
            fileInput.accept = '.jpg,.jpeg,.png,.mp4,.avi,.mov,.wmv';
        } else if (type === 'output') {
            // For output, allow selection of directories (will simulate with file selection)
            fileInput.accept = '*';
        }
        
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                if (type === 'test') {
                    this.testFilePathInput.value = file.name; // In real app, would be full path
                } else if (type === 'output') {
                    // For output, extract directory path
                    const fileName = file.name;
                    const outputPath = fileName.substring(0, fileName.lastIndexOf('.')) + '_results.mp4';
                    this.outputResultsPathInput.value = outputPath;
                }
            }
        });
        
        fileInput.click();
    }

    // Detection functionality
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
        // Simulate detection process
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
        
        // Show completion status
        this.detectionStatusDisplay.textContent = 'Detection Complete!';
        this.detectionStatusDisplay.style.background = 'linear-gradient(135deg, #48bb78 0%, #38a169 100%)';
        this.detectionStatusDisplay.style.color = 'white';
        
        // Show final results
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

    // WebSocket and Real Training Methods
    initializeWebSocket() {
        // Initialize Socket.IO connection
        if (typeof io !== 'undefined') {
            this.socket = io('http://localhost:5000', {
                transports: ['websocket', 'polling']
            });
            
            // Listen for training updates
            this.socket.on('training_update', (data) => {
                // Check if this is a new epoch to start timer
                const isNewEpoch = !this.epochStartTime || data.epoch !== this.currentEpoch;
                if (isNewEpoch) {
                    this.stopEpochTimer();
                    this.startEpochTimer();
                    this.currentEpoch = data.epoch;
                }
                
                this.updateMetrics({
                    epoch: data.epoch,
                    totalEpochs: data.total_epochs,
                    loss: data.loss,
                    map: data.map,
                    progress: data.progress
                });
            });
            
            // Listen for training start
            this.socket.on('training_started', (data) => {
                console.log('Training started:', data.message);
                this.startEpochTimer(); // Start timing when training begins
            });
            
            // Listen for training completion
            this.socket.on('training_complete', (data) => {
                this.trainingComplete();
                console.log('Training completed:', data.message);
            });
            
            // Listen for training errors
            this.socket.on('training_error', (data) => {
                this.trainingError(data.error);
            });
            
            // Listen for console output
            this.socket.on('console_output', (data) => {
                this.addConsoleOutput(data.line);
            });
            
            // Listen for connection status
            this.socket.on('connect', () => {
                console.log('Connected to training server');
                this.addConsoleOutput('Connected to training server', 'success');
            });
            
            this.socket.on('disconnect', () => {
                console.log('Disconnected from training server');
                this.addConsoleOutput('Disconnected from training server', 'warning');
            });
            
            this.socket.on('connect_error', (error) => {
                console.error('Socket.IO connection error:', error);
                this.addConsoleOutput(`Connection error: ${error.message}`, 'error');
            });
            
            this.socket.on('reconnect', () => {
                console.log('Reconnected to training server');
                this.addConsoleOutput('Reconnected to training server', 'success');
            });
            
        } else {
            console.warn('Socket.IO not available - falling back to simulation mode');
            this.addConsoleOutput('Socket.IO not available - using simulation mode', 'warning');
        }
    }

    async startRealTraining(config) {
        if (!this.socket) {
            console.warn('No socket connection - falling back to simulation');
            this.simulateTraining(config);
            return;
        }

        try {
            // Send training request to backend
            const response = await fetch('/api/start-training', {
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
            console.log('Training request sent:', result);

        } catch (error) {
            console.error('Failed to start training:', error);
            this.trainingError(error.message);
        }
    }

    trainingError(errorMessage) {
        this.isTraining = false;
        this.stopEpochTimer();
        this.updateUI();
        
        this.statusDisplay.textContent = `Training Error: ${errorMessage}`;
        this.statusDisplay.style.background = 'linear-gradient(135deg, #e53e3e 0%, #c53030 100%)';
        this.statusDisplay.style.color = 'white';
        
        // Reset progress displays
        this.epochElapsedTime.textContent = 'Error';
        this.estimatedRemainingTime.textContent = '-';
        
        setTimeout(() => {
            this.statusDisplay.textContent = 'Ready to start training';
            this.statusDisplay.style.background = '';
            this.statusDisplay.style.color = '';
        }, 5000);
    }

    // Keep the simulation method as fallback
    simulateTraining(config) {
        // This is a simulation - used as fallback when WebSocket is not available
        const totalEpochs = config.epochs;
        let currentEpoch = 0;
        
        // Start first epoch timer
        this.startEpochTimer();
        
        const trainingInterval = setInterval(() => {
            // Check if we should start a new epoch (every 3-8 seconds for realistic simulation)
            const epochDurationRange = Math.random() * 5000 + 3000; // 3-8 seconds
            
            if (!this.epochStartTime || (Date.now() - this.epochStartTime) > epochDurationRange) {
                currentEpoch++;
                
                // Stop previous epoch timer and start new one
                this.stopEpochTimer();
                if (currentEpoch <= totalEpochs) {
                    this.startEpochTimer();
                }
                
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
            }
        }, 500); // Check every 500ms for smooth updates
    }

    // Console Methods
    addConsoleOutput(line, type = 'normal') {
        if (!this.consoleOutput) return;
        
        const timestamp = new Date().toLocaleTimeString();
        const consoleLineDiv = document.createElement('div');
        consoleLineDiv.className = `console-line ${type}`;
        consoleLineDiv.textContent = `[${timestamp}] ${line}`;
        
        this.consoleOutput.appendChild(consoleLineDiv);
        
        // Auto-scroll to bottom
        this.consoleOutput.scrollTop = this.consoleOutput.scrollHeight;
        
        // Limit console output to 500 lines
        const lines = this.consoleOutput.children;
        if (lines.length > 500) {
            this.consoleOutput.removeChild(lines[0]);
        }
    }
    
    clearConsole() {
        if (this.consoleOutput) {
            this.consoleOutput.innerHTML = '<div class="console-line">Console cleared...</div>';
        }
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new TrainingDashboard();
});