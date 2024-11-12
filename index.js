const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let pythonProcess;

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'frontend', 'preloader.js'),
      nodeIntegration: true, // Enable Node.js integration
      contextIsolation: false, // Disable context isolation
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

function startPythonProcess() {
  pythonProcess = spawn('python', [path.join(__dirname, 'backend', 'main.py')]);

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
  startPythonProcess(); // Start the Python process before creating the window
  createWindow();

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

// Read the file and send the content to the renderer process
ipcMain.handle('read-file', async (event, filePath) => {
  try {
    const data = fs.readFileSync(filePath, 'utf8');
    return data;
  } catch (err) {
    console.error('Error reading file:', err);
    throw err;
  }
});