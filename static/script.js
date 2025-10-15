document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const status = document.getElementById("status");
    const rtsp = document.getElementById("rtsp");
    const tgToken = document.getElementById("tg_token");
    const tgChat = document.getElementById("tg_chat");
    const startHour = document.getElementById("start_hour");
    const endHour = document.getElementById("end_hour");
    const minConf = document.getElementById("min_conf");
    const confValue = document.getElementById("conf_value");
    const video = document.getElementById("video");
    const loading = document.getElementById("loading");

    // Load saved configuration from localStorage
    loadSavedConfig();

    // Populate hour selectors
    for (let i = 0; i < 24; i++) {
        const option1 = document.createElement("option");
        option1.value = i;
        option1.text = `${i.toString().padStart(2, '0')}:00`;
        startHour.appendChild(option1);

        const option2 = document.createElement("option");
        option2.value = i;
        option2.text = `${i.toString().padStart(2, '0')}:00`;
        endHour.appendChild(option2);
    }
    
    // Option 24 for endHour
    const option24 = document.createElement("option");
    option24.value = 24;
    option24.text = "24:00";
    endHour.appendChild(option24);

    // Server values or defaults
    const configStartHour = "{{ config.start_hour if config else 0 }}";
    const configEndHour = "{{ config.end_hour if config else 24 }}";
    startHour.value = configStartHour || "0";
    endHour.value = configEndHour || "24";

    // Update confidence display
    minConf.addEventListener("input", () => {
        const val = parseFloat(minConf.value);
        confValue.textContent = val.toFixed(2);
        confValue.parentElement.querySelector("span:last-child").textContent = 
            ` (${Math.round(val * 100)}%)`;
    });

    // Start Button
    startBtn.onclick = async () => {
        // Basic validation
        if (!rtsp.value && rtsp.value !== "0") {
            alert("⚠️ Please enter an RTSP URL or '0' for webcam");
            return;
        }

        if (!tgToken.value.trim()) {
            const confirm = window.confirm(
                "⚠️ You haven't configured the Telegram Bot.\n" +
                "The system will work but will NOT send notifications.\n\n" +
                "Do you want to continue?"
            );
            if (!confirm) return;
        }

        // Save configuration
        saveConfig();

        const payload = {
            rtsp_url: rtsp.value.trim() || "0",
            telegram_token: tgToken.value.trim(),
            telegram_chat_id: tgChat.value.trim(),
            start_hour: parseInt(startHour.value),
            end_hour: parseInt(endHour.value),
            min_conf: parseFloat(minConf.value)
        };

        try {
            updateStatus("⏳ Starting system...", "");
            loading.classList.add("active");
            startBtn.disabled = true;

            const response = await fetch("/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (response.ok) {
                updateStatus("🟢 System running", "status-running");
                video.style.display = "block";
                
                // Reload video image after a moment
                setTimeout(() => {
                    video.src = "/video_feed?" + new Date().getTime();
                    loading.classList.remove("active");
                }, 2000);
            } else {
                throw new Error(data.error || "Unknown error");
            }
        } catch (error) {
            updateStatus("❌ Error: " + error.message, "status-stopped");
            alert("Error starting system:\n" + error.message);
            loading.classList.remove("active");
        } finally {
            startBtn.disabled = false;
        }
    };

    // Stop Button
    stopBtn.onclick = async () => {
        try {
            updateStatus("⏳ Stopping system...", "");
            stopBtn.disabled = true;

            const response = await fetch("/stop", { method: "POST" });
            const data = await response.json();

            updateStatus("🔴 System stopped", "status-stopped");
            loading.classList.remove("active");
        } catch (error) {
            updateStatus("❌ Error stopping", "status-stopped");
            alert("Error stopping system:\n" + error.message);
        } finally {
            stopBtn.disabled = false;
        }
    };

    // Helper functions
    function updateStatus(message, className) {
        status.textContent = message;
        status.className = className;
    }

    function saveConfig() {
        const config = {
            rtsp: rtsp.value,
            tgToken: tgToken.value,
            tgChat: tgChat.value,
            startHour: startHour.value,
            endHour: endHour.value,
            minConf: minConf.value
        };
        localStorage.setItem("detectorConfig", JSON.stringify(config));
    }

    function loadSavedConfig() {
        try {
            const saved = localStorage.getItem("detectorConfig");
            if (saved) {
                const config = JSON.parse(saved);
                rtsp.value = config.rtsp || "";
                tgToken.value = config.tgToken || "";
                tgChat.value = config.tgChat || "";
                startHour.value = config.startHour || "0";
                endHour.value = config.endHour || "24";
                minConf.value = config.minConf || "0.45";
                
                // Trigger event to update display
                minConf.dispatchEvent(new Event("input"));
            }
        } catch (e) {
            console.error("Error loading saved configuration:", e);
        }
    }

    // Handle image loading errors
    video.onerror = () => {
        console.log("Error loading video feed");
    };

    // Initialize confidence display
    minConf.dispatchEvent(new Event("input"));
});
