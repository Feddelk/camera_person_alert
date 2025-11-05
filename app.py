from flask import Flask, render_template, Response, request, jsonify
from camera import VideoCamera, DetectorConfig
import threading, time, os, sys

# Load configuration - REQUIRED
#####

try:
    import config
except ImportError:
    print("\n" + "="*60)
    print("❌ ERROR: config.py not found!")
    print("="*60)
    print("Please create config.py with your settings.")
    print("See README.md for instructions.")
    print("="*60 + "\n")
    sys.exit(1)

app = Flask(__name__)

# Flask configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Global camera manager
camera_manager = {"camera": None, "lock": threading.Lock()}

@app.route("/")
def index():
    """Main page"""
    config_data = {
        'rtsp_url': config.RTSP_URL,
        'telegram_token': config.TELEGRAM_BOT_TOKEN,
        'telegram_chat_id': config.TELEGRAM_CHAT_ID,
        'start_hour': config.NOTIFY_START_HOUR,
        'end_hour': config.NOTIFY_END_HOUR,
        'min_conf': config.MIN_CONFIDENCE
    }
    return render_template("index_simple.html", config=config_data)

def gen_frames():
    """Frame generator for video streaming"""
    no_camera_shown = False
    
    while True:
        with camera_manager["lock"]:
            cam = camera_manager["camera"]
            
            if not cam:
                if not no_camera_shown:
                    print("[Stream] ⏸️ No active camera")
                    no_camera_shown = True
                time.sleep(0.1)
                continue
            
            no_camera_shown = False
            frame = cam.get_frame()
        
        if frame is None:
            time.sleep(0.01)
            continue
        
        # Enviar frame en formato MJPEG sin delay
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video_feed")
def video_feed():
    """Video stream endpoint"""
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/status")
def get_status():
    """Get current system status"""
    with camera_manager["lock"]:
        is_running = camera_manager["camera"] is not None
    
    return jsonify({
        "running": is_running
    })

def cleanup():
    """Clean up resources on application close"""
    print("\n[App] 🧹 Cleaning up resources...")
    with camera_manager["lock"]:
        if camera_manager["camera"]:
            camera_manager["camera"].stop()
            camera_manager["camera"] = None
    print("[App] ✓ Cleanup completed\n")

import atexit
atexit.register(cleanup)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎥 PERSON DETECTOR - RTSP + TELEGRAM")
    print("="*60)
    print("🚀 AUTO-START mode (config.py)")
    print("="*60)
    print(f"📹 Camera: {config.RTSP_URL}")
    print(f"🤖 Telegram: {'Configured ✓' if config.TELEGRAM_BOT_TOKEN else 'Not configured ⚠️'}")
    print(f"⏰ Schedule: {config.NOTIFY_START_HOUR}:00 - {config.NOTIFY_END_HOUR}:00")
    print(f"🎯 Confidence: {config.MIN_CONFIDENCE}")
    print(f"🌐 Server: http://localhost:{config.PORT}")
    print("="*60 + "\n")
    
    cfg = DetectorConfig(
        rtsp_url=config.RTSP_URL,
        telegram_bot_token=config.TELEGRAM_BOT_TOKEN,
        telegram_chat_id=config.TELEGRAM_CHAT_ID,
        notify_start_hour=config.NOTIFY_START_HOUR,
        notify_end_hour=config.NOTIFY_END_HOUR,
        min_confidence=config.MIN_CONFIDENCE
    )
    
    with camera_manager["lock"]:
        camera_manager["camera"] = VideoCamera(cfg)
        camera_manager["camera"].start()
    
    print("✅ Camera started automatically")
    print("🌐 Open your browser at: http://localhost:{}\n".format(config.PORT))
    
    try:
        app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[App] 👋 Closing server...")
        cleanup()
        sys.exit(0)
