/**
 * Offline Smart Assistant JavaScript
 * Manages real-time communication with the assistant backend
 */

class SmartAssistant {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.isListening = false;
        this.cameraActive = false;
        this.currentLanguage = 'en';
        
        // DOM elements
        this.elements = {};
        this.initializeElements();
        
        // Event listeners
        this.setupEventListeners();
        
        // Initialize connection
        this.initializeSocket();
        
        console.log('🤖 Smart Assistant initialized');
    }
    
    initializeElements() {
        // Get all DOM elements
        this.elements = {
            statusBadge: document.getElementById('status-badge'),
            speechStatus: document.getElementById('speech-status'),
            cameraStatus: document.getElementById('camera-status'),
            aiStatus: document.getElementById('ai-status'),
            emotionStatus: document.getElementById('emotion-status'),
            
            micIcon: document.getElementById('mic-icon'),
            cameraIcon: document.getElementById('camera-icon'),
            aiIcon: document.getElementById('ai-icon'),
            emotionIcon: document.getElementById('emotion-icon'),
            
            startListeningBtn: document.getElementById('start-listening-btn'),
            startCameraBtn: document.getElementById('start-camera-btn'),
            toggleCameraBtn: document.getElementById('toggle-camera-btn'),
            languageSelect: document.getElementById('language-select'),
            textInput: document.getElementById('text-input'),
            sendTextBtn: document.getElementById('send-text-btn'),
            
            conversationLog: document.getElementById('conversation-log'),
            audioFeedback: document.getElementById('audio-feedback'),
            
            describeSceneBtn: document.getElementById('describe-scene-btn'),
            checkEmotionsBtn: document.getElementById('check-emotions-btn'),
            getStatusBtn: document.getElementById('get-status-btn'),
            stopAllBtn: document.getElementById('stop-all-btn')
        };
    }
    
    setupEventListeners() {
        // Control buttons
        this.elements.startListeningBtn.addEventListener('click', () => this.toggleListening());
        this.elements.startCameraBtn.addEventListener('click', () => this.toggleCamera());
        this.elements.toggleCameraBtn.addEventListener('click', () => this.toggleCameraMode());
        this.elements.sendTextBtn.addEventListener('click', () => this.sendTextMessage());
        
        // Language selection
        this.elements.languageSelect.addEventListener('change', (e) => this.switchLanguage(e.target.value));
        
        // Text input
        this.elements.textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendTextMessage();
            }
        });
        
        // Quick action buttons
        this.elements.describeSceneBtn.addEventListener('click', () => this.requestSceneDescription());
        this.elements.checkEmotionsBtn.addEventListener('click', () => this.checkEmotions());
        this.elements.getStatusBtn.addEventListener('click', () => this.getStatus());
        this.elements.stopAllBtn.addEventListener('click', () => this.stopAll());
        
        // Keyboard shortcuts for accessibility
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
    }
    
    initializeSocket() {
        try {
            this.socket = io();
            
            this.socket.on('connect', () => {
                console.log('✅ Connected to assistant');
                this.isConnected = true;
                this.updateStatus('Connected', 'success');
                this.announceToScreenReader('Connected to assistant');
            });
            
            this.socket.on('disconnect', () => {
                console.log('❌ Disconnected from assistant');
                this.isConnected = false;
                this.updateStatus('Disconnected', 'danger');
                this.announceToScreenReader('Disconnected from assistant');
            });
            
            this.socket.on('assistant_message', (data) => this.handleAssistantMessage(data));
            this.socket.on('speech_recognized', (data) => this.handleSpeechRecognized(data));
            this.socket.on('status_update', (data) => this.handleStatusUpdate(data));
            this.socket.on('camera_status', (data) => this.handleCameraStatus(data));
            this.socket.on('listening_status', (data) => this.handleListeningStatus(data));
            this.socket.on('camera_mode_changed', (data) => this.handleCameraModeChanged(data));
            this.socket.on('language_switched', (data) => this.handleLanguageSwitched(data));
            this.socket.on('error', (data) => this.handleError(data));
            
        } catch (error) {
            console.error('❌ Socket initialization failed:', error);
            this.updateStatus('Connection Failed', 'danger');
        }
    }
    
    // Socket event handlers
    handleAssistantMessage(data) {
        console.log('🗣️ Assistant message:', data);
        this.addMessageToLog('assistant', data.text, data.language, data.emotion);
        this.announceToScreenReader(`Assistant says: ${data.text}`);
        
        // Update emotion status if provided
        if (data.emotion) {
            this.updateEmotionStatus(data.emotion);
        }
    }
    
    handleSpeechRecognized(data) {
        console.log('🎤 Speech recognized:', data);
        this.addMessageToLog('user', data.text, data.language);
        this.announceToScreenReader(`You said: ${data.text}`);
    }
    
    handleStatusUpdate(data) {
        console.log('📊 Status update:', data);
        this.updateSystemStatus(data);
    }
    
    handleCameraStatus(data) {
        this.cameraActive = data.active;
        this.updateCameraStatus(data.active, data.mode);
        
        // Update button states
        this.elements.toggleCameraBtn.disabled = !data.active;
        this.elements.startCameraBtn.textContent = data.active ? 'Stop Camera' : 'Start Camera';
        this.elements.startCameraBtn.className = data.active ? 
            'btn btn-outline-danger btn-lg w-100' : 
            'btn btn-outline-primary btn-lg w-100';
    }
    
    handleListeningStatus(data) {
        this.isListening = data.active;
        this.updateSpeechStatus(data.active);
        
        // Update button states
        this.elements.startListeningBtn.textContent = data.active ? 'Stop Listening' : 'Start Listening';
        this.elements.startListeningBtn.className = data.active ? 
            'btn btn-outline-danger btn-lg w-100' : 
            'btn btn-outline-success btn-lg w-100';
        
        this.announceToScreenReader(data.active ? 'Speech recognition started' : 'Speech recognition stopped');
    }
    
    handleCameraModeChanged(data) {
        console.log('📷 Camera mode changed:', data.mode);
        this.updateCameraStatus(this.cameraActive, data.mode);
        this.announceToScreenReader(`Camera switched to ${data.mode} mode`);
    }
    
    handleLanguageSwitched(data) {
        console.log('🌐 Language switched:', data);
        if (data.success) {
            this.currentLanguage = data.language;
            this.announceToScreenReader(`Language switched to ${this.getLanguageName(data.language)}`);
        } else {
            this.announceToScreenReader('Failed to switch language');
        }
    }
    
    handleError(data) {
        console.error('❌ Assistant error:', data);
        this.addMessageToLog('system', `Error: ${data.message}`, 'en');
        this.announceToScreenReader(`System error: ${data.message}`);
    }
    
    // UI update methods
    updateStatus(status, type = 'secondary') {
        this.elements.statusBadge.textContent = status;
        this.elements.statusBadge.className = `badge bg-${type} ms-2`;
    }
    
    updateSpeechStatus(active) {
        this.elements.speechStatus.textContent = active ? 'Listening...' : 'Off';
        this.elements.speechStatus.className = active ? 'text-success' : 'text-muted';
        this.elements.micIcon.className = active ? 'fas fa-microphone text-success status-icon' : 'fas fa-microphone-slash text-muted status-icon';
    }
    
    updateCameraStatus(active, mode = 'scene') {
        if (active) {
            this.elements.cameraStatus.textContent = `Active (${mode})`;
            this.elements.cameraStatus.className = 'text-success';
            this.elements.cameraIcon.className = 'fas fa-camera text-success status-icon';
        } else {
            this.elements.cameraStatus.textContent = 'Off';
            this.elements.cameraStatus.className = 'text-muted';
            this.elements.cameraIcon.className = 'fas fa-camera text-muted status-icon';
        }
    }
    
    updateAIStatus(status) {
        this.elements.aiStatus.textContent = status;
        const isReady = status.toLowerCase().includes('ready') || status.toLowerCase().includes('connected');
        this.elements.aiStatus.className = isReady ? 'text-success' : 'text-warning';
        this.elements.aiIcon.className = isReady ? 'fas fa-brain text-success status-icon' : 'fas fa-brain text-warning status-icon';
    }
    
    updateEmotionStatus(emotion) {
        this.elements.emotionStatus.textContent = emotion;
        
        // Color based on emotion
        const emotionColors = {
            'Happy': 'text-success',
            'Sad': 'text-info',
            'Angry': 'text-danger',
            'Fear': 'text-warning',
            'Surprise': 'text-warning',
            'Neutral': 'text-muted'
        };
        
        this.elements.emotionStatus.className = emotionColors[emotion] || 'text-muted';
        
        // Icon based on emotion
        const emotionIcons = {
            'Happy': 'fas fa-smile',
            'Sad': 'fas fa-frown',
            'Angry': 'fas fa-angry',
            'Fear': 'fas fa-surprise',
            'Surprise': 'fas fa-surprise',
            'Neutral': 'fas fa-meh'
        };
        
        const iconClass = emotionIcons[emotion] || 'fas fa-heart';
        this.elements.emotionIcon.className = `${iconClass} ${emotionColors[emotion] || 'text-muted'} status-icon`;
    }
    
    updateSystemStatus(statusData) {
        // Update AI status
        if (statusData.components && statusData.components.gemma) {
            const gemmaStatus = statusData.components.gemma;
            if (gemmaStatus.model_ready) {
                this.updateAIStatus('Ready');
            } else {
                this.updateAIStatus('Loading...');
            }
        }
        
        // Update other statuses as needed
        if (statusData.user_context && statusData.user_context.current_emotion) {
            this.updateEmotionStatus(statusData.user_context.current_emotion);
        }
    }
    
    addMessageToLog(sender, message, language = 'en', emotion = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const timestamp = new Date().toLocaleTimeString();
        const senderIcon = sender === 'assistant' ? 'fas fa-robot' : 
                          sender === 'user' ? 'fas fa-user' : 'fas fa-exclamation-triangle';
        const senderName = sender === 'assistant' ? 'Assistant' : 
                          sender === 'user' ? 'You' : 'System';
        
        let emotionBadge = '';
        if (emotion && emotion !== 'Neutral') {
            emotionBadge = `<span class="badge bg-secondary ms-2">${emotion}</span>`;
        }
        
        messageDiv.innerHTML = `
            <div class="message-header">
                <i class="${senderIcon} me-2" aria-hidden="true"></i>
                <strong>${senderName}</strong>
                ${emotionBadge}
                <span class="timestamp">${timestamp}</span>
            </div>
            <div class="message-content">${this.escapeHtml(message)}</div>
        `;
        
        this.elements.conversationLog.appendChild(messageDiv);
        
        // Auto-scroll to bottom
        messageDiv.scrollIntoView({ behavior: 'smooth' });
        
        // Keep only last 50 messages
        while (this.elements.conversationLog.children.length > 50) {
            this.elements.conversationLog.removeChild(this.elements.conversationLog.firstChild);
        }
    }
    
    // Action methods
    toggleListening() {
        if (!this.isConnected) {
            this.announceToScreenReader('Not connected to assistant');
            return;
        }
        
        if (this.isListening) {
            this.socket.emit('stop_listening');
        } else {
            this.socket.emit('start_listening');
        }
    }
    
    toggleCamera() {
        if (!this.isConnected) {
            this.announceToScreenReader('Not connected to assistant');
            return;
        }
        
        if (this.cameraActive) {
            this.socket.emit('stop_camera');
        } else {
            this.socket.emit('start_camera', { mode: 'scene' });
        }
    }
    
    toggleCameraMode() {
        if (!this.isConnected || !this.cameraActive) {
            this.announceToScreenReader('Camera must be active to switch modes');
            return;
        }
        
        this.socket.emit('toggle_camera_mode');
    }
    
    sendTextMessage() {
        const text = this.elements.textInput.value.trim();
        if (!text) return;
        
        if (!this.isConnected) {
            this.announceToScreenReader('Not connected to assistant');
            return;
        }
        
        this.socket.emit('send_text', {
            text: text,
            language: this.currentLanguage
        });
        
        this.elements.textInput.value = '';
    }
    
    switchLanguage(language) {
        if (!this.isConnected) {
            this.announceToScreenReader('Not connected to assistant');
            return;
        }
        
        this.socket.emit('switch_language', { language: language });
    }
    
    requestSceneDescription() {
        if (!this.isConnected) {
            this.announceToScreenReader('Not connected to assistant');
            return;
        }
        
        this.socket.emit('request_scene_description');
        this.announceToScreenReader('Requesting scene description');
    }
    
    checkEmotions() {
        if (!this.isConnected) {
            this.announceToScreenReader('Not connected to assistant');
            return;
        }
        
        // Switch to emotion mode and request analysis
        this.socket.emit('start_camera', { mode: 'emotion' });
        this.announceToScreenReader('Checking emotions');
    }
    
    getStatus() {
        if (!this.isConnected) {
            this.announceToScreenReader('Not connected to assistant');
            return;
        }
        
        this.socket.emit('get_status');
        this.announceToScreenReader('Getting system status');
    }
    
    stopAll() {
        if (!this.isConnected) {
            this.announceToScreenReader('Not connected to assistant');
            return;
        }
        
        this.socket.emit('stop_listening');
        this.socket.emit('stop_camera');
        this.announceToScreenReader('Stopping all services');
    }
    
    // Accessibility methods
    announceToScreenReader(message) {
        this.elements.audioFeedback.textContent = message;
        // Clear after a delay so it doesn't accumulate
        setTimeout(() => {
            this.elements.audioFeedback.textContent = '';
        }, 3000);
    }
    
    handleKeyboardShortcuts(event) {
        // Only handle shortcuts when not typing in input fields
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA' || event.target.tagName === 'SELECT') {
            return;
        }
        
        if (event.ctrlKey || event.metaKey) {
            switch (event.key) {
                case 'm': // Ctrl+M - Toggle microphone
                    event.preventDefault();
                    this.toggleListening();
                    break;
                case 'c': // Ctrl+C - Toggle camera
                    event.preventDefault();
                    this.toggleCamera();
                    break;
                case 's': // Ctrl+S - Describe scene
                    event.preventDefault();
                    this.requestSceneDescription();
                    break;
                case 'e': // Ctrl+E - Check emotions
                    event.preventDefault();
                    this.checkEmotions();
                    break;
            }
        }
    }
    
    // Utility methods
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    getLanguageName(code) {
        const names = {
            'en': 'English',
            'hi': 'Hindi',
            'gu': 'Gujarati'
        };
        return names[code] || code;
    }
}

// Initialize the assistant when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.smartAssistant = new SmartAssistant();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden && window.smartAssistant) {
        // Page is hidden, pause non-essential services
        console.log('🔇 Page hidden, maintaining core services');
    } else if (window.smartAssistant) {
        // Page is visible again
        console.log('👁️ Page visible, resuming services');
    }
});

// Handle before unload
window.addEventListener('beforeunload', () => {
    if (window.smartAssistant && window.smartAssistant.socket) {
        window.smartAssistant.socket.disconnect();
    }
});
