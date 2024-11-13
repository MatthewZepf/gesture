import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import imutils
import websockets
import asyncio
import base64
import os
import json
from config import Config
from utils import process_frame

config = Config()

async def send_frames(websocket, config=config):
    cap = cv2.VideoCapture(0)
    previous_landmarks = None

    while not config.shutdown:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Process the frame
        if ret:
            frame = imutils.resize(frame, width=720)
            previous_landmarks = process_frame(frame, previous_landmarks, config)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Send the frame over the WebSocket
            await websocket.send(frame_base64)

    # When everything is done, release the capture
    cap.release()

async def handle_commands(websocket):
    async for message in websocket:
        data = json.loads(message)
        if data['command'] == 'shutdown':
            config.shutdown = True
            break

async def handler(websocket, path):
    print("Client connected")
    await asyncio.gather(
        send_frames(websocket),
        handle_commands(websocket)
    )

async def main():
    
    server = await websockets.serve(handler, "localhost", 0)
    # write socket port to local file
    with open('socket_port.txt', 'w') as f:
        f.write(str(server.sockets[0].getsockname()[1]))
    await server.wait_closed()
    os.remove('socket_port.txt')

if __name__ == "__main__":
    asyncio.run(main())