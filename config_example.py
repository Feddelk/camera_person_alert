# Person Detection System Configuration
# Copy this file to config.py and edit with your values

# ==================== CAMERA ====================
RTSP_URL = "rtsp://username:password@192.168.1.100:554/stream"  # Your RTSP camera or "0" for webcam

# ==================== TELEGRAM ====================
TELEGRAM_BOT_TOKEN = ""  # Get from @BotFather on Telegram
TELEGRAM_CHAT_ID = ""    # Get from @userinfobot on Telegram

# ==================== DETECTION ====================
MIN_CONFIDENCE = 0.45        # 0.3 to 0.9 (recommended: 0.45)
YOLO_MODEL = "yolov8n.pt"   # yolov8n.pt (fast) or yolov8s.pt (accurate)

# ==================== NOTIFICATIONS ====================
NOTIFY_START_HOUR = 0       # Start hour (0-23) in your LOCAL timezone
NOTIFY_END_HOUR = 24        # End hour (0-24) in your LOCAL timezone
COOLDOWN_SECONDS = 15       # Seconds between notifications
UTC_OFFSET = 0              # Your UTC offset (e.g., -3 for Argentina, +9 for Japan)

# ==================== SERVER ====================
HOST = "0.0.0.0"           # 0.0.0.0 for remote access, 127.0.0.1 local only
PORT = 5000                # Web server port

# ==================== ADVANCED ====================
FRAME_SKIP = 3             # Process 1 out of N frames (higher = faster)
DETECTION_SIZE = 320       # Resolution for detection (lower = faster)
DISPLAY_SIZE = 640         # Resolution for display
JPEG_QUALITY = 75          # Stream quality (60-95)
MAX_CAPTURES = 50          # Maximum captures to keep
USE_GPU = False            # True to use GPU (requires CUDA)
