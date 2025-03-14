import os
import cv2
import numpy as np
import dlib
from imutils import face_utils
import RPi.GPIO as GPIO
import time
import json  # Ensure json module is imported
import paho.mqtt.client as mqtt

# Set Qt platform to avoid errors when running OpenCV on Raspberry Pi
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Disable GPIO warnings
GPIO.setwarnings(False)

# Define GPIO pin for LED
LED_PIN = 23

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

# Initialize the camera and face detection modules
cap = cv2.VideoCapture(0)  # Open webcam
hog_face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/pi/Downloads/drowsinessDetector-master/shape_predictor_68_face_landmarks.dat")

# MQTT Broker Details (Local Mosquitto)
MQTT_BROKER = "localhost"  # Local broker on the Raspberry Pi
MQTT_PORT = 1883  # Default MQTT port (no TLS)
MQTT_TOPIC = "status"  # MQTT topic to publish to

# Initialize MQTT Client
mqtt_client = mqtt.Client()

def connect_mqtt():
    """Connect to the local MQTT broker."""
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        print("Connected to MQTT Broker!")
    except Exception as e:
        print(f"Failed to connect to MQTT Broker: {e}")

def publish_status(status, led_state):
    """Publish driver status and LED state to the MQTT topic."""
    payload = {
        "status": status,
        "led": led_state,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    json_payload = json.dumps(payload)  # Convert dictionary to valid JSON string
    print(f"Publishing: {json_payload}")  # Log the JSON payload
    try:
        mqtt_client.publish(MQTT_TOPIC, json_payload)  # Send JSON string
        print(f"Published: {json_payload}")
    except Exception as e:
        print(f"Failed to publish status: {e}")

# Connect to MQTT Broker
connect_mqtt()

# State variables
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

def compute(ptA, ptB):
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    """Check if eyes are closed based on aspect ratio."""
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2  # Eyes open
    elif 0.21 < ratio <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Eyes closed

def led_on():
    """Turn the LED on."""
    GPIO.output(LED_PIN, GPIO.HIGH)
    return "ON"

def led_off():
    """Turn the LED off."""
    GPIO.output(LED_PIN, GPIO.LOW)
    return "OFF"

# Main loop with graceful exit
try:
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = hog_face_detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if left_blink == 0 or right_blink == 0:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 6:
                    status = "SLEEPING !!!"
                    color = (0, 0, 255)
                    led_state = led_on()
                    publish_status(status, led_state)

            elif left_blink == 1 or right_blink == 1:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:
                    status = "DROWSY !"
                    color = (0, 0, 255)
                    led_state = led_on()
                    publish_status(status, led_state)

            else:
                sleep = 0
                drowsy = 0
                active += 1
                if active > 6:
                    status = "ACTIVE :)"
                    color = (0, 255, 0)
                    led_state = led_off()
                    publish_status(status, led_state)

            cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        cv2.imshow("Driver Drowsiness Detection", frame)

        # Check for exit key (ESC)
        if cv2.waitKey(1) == 27:  # ESC key
            break

except KeyboardInterrupt:
    print("Program interrupted. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()  # Cleanup GPIO pins
    mqtt_client.disconnect()  # Disconnect from MQTT broker
