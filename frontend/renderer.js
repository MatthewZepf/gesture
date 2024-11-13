const { ipcRenderer } = require('electron');

// Request the file content from the main process
ipcRenderer.invoke('read-file', 'socket_port.txt').then((port) => {
  console.log('WebSocket port:', port);
  const ws = new WebSocket('ws://localhost:' + port);

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
    console.log('Disconnected from WebSocket server');
  };

});

// Create an image element in the DOM to display the frames
const imgElement = document.createElement('img');
imgElement.id = 'frame';
document.body.appendChild(imgElement);
