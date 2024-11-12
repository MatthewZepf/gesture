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

  document.getElementById('start').addEventListener('click', () => {
    const message = JSON.stringify({ type: 'command', command: 'start' });
    ws.send(message);
    console.log('Sent start command');
  });

  document.getElementById('stop').addEventListener('click', () => {
    const message = JSON.stringify({ type: 'command', command: 'stop' });
    ws.send(message);
    console.log('Sent stop command');
  });

  document.getElementById('send-settings').addEventListener('click', () => {
    const settings = {
      setting1: 'value1',
      setting2: 'value2'
    };
    const message = JSON.stringify({ type: 'command', command: 'settings', settings: settings });
    ws.send(message);
    console.log('Sent settings');
  });

  document.getElementById('shutdown').addEventListener('click', () => {
    const message = JSON.stringify({ type: 'command', command: 'shutdown' });
    ws.send(message);
    console.log('Sent shutdown command');
  });
}).catch((err) => {
  console.error('Error reading file:', err);
});

// Create an image element in the DOM to display the frames
const imgElement = document.createElement('img');
imgElement.id = 'frame';
document.body.appendChild(imgElement);
