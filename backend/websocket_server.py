import asyncio
import websockets

async def echo(websocket, path):
    async for message in websocket:
        print("Received message")
        # Process the received message (e.g., save the frame, display it, etc.)
        await websocket.send("Message received")

def start_websocket_server():
    start_server = websockets.serve(echo, "localhost", 8000)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    start_websocket_server()