const ws = new WebSocket('ws://localhost:8000');

// Create an image element in the DOM to display the frames
const imgElement = document.createElement('img');
document.body.appendChild(imgElement);

ws.onopen = () => {
    console.log('Connected to WebSocket server');
};

ws.onmessage = (event) => {
    // Get the base64 frame from the backend
    const frameBase64 = event.data;

    // Set the image source to the base64 data
    imgElement.src = 'data:image/jpeg;base64,' + frameBase64;
};

ws.onclose = () => {
    console.log('WebSocket connection closed');
};
