const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let pythonProcess;

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'frontend', 'preloader.js'),
    },
  });

  win.loadFile(path.join(__dirname, 'frontend', 'index.html'));

  // Open Developer Tools
  win.webContents.openDevTools();
}

function killPythonProcess() {
  if (pythonProcess) {
    pythonProcess.kill();
  }
}

function startPythonWebSocketServer() {
  pythonProcess = spawn('python', [path.join(__dirname, 'backend', 'websocket_server.py')]);

  pythonProcess.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
  });
}

app.whenReady().then(() => {
  createWindow();
  startPythonWebSocketServer(); // Start the WebSocket server when the app is ready

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  killPythonProcess();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  killPythonProcess();
});