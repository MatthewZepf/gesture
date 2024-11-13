import React, { useState } from "react";
import "./style.css"; // Ensure styles are updated here
import { useNavigate } from "react-router-dom";

const InProgress = () => {
  const [clickedButtons, setClickedButtons] = useState({
    settings: false,
    practiceMode: false,
    toggleCamera: true,
  });
  const navigate = useNavigate();

  const handleButtonClick = (buttonName) => {
    setClickedButtons((prev) => ({
      ...prev,
      [buttonName]: !prev[buttonName],
    }));
  };

  const handleStop = () => {
    console.log("STOPPING");
    navigate("/");
  };

  return (
    <div className="in-progress-container">
      
        <div 
          className="top-left-image-container" 
          onClick={handleStop}
          role="button" 
          aria-label="Stop"
          tabIndex={0}
        >
          <img src="./logo.png" alt="Top Left" className="top-left-image" />
        </div>
      <div className="content">
        <div className="top-bar">
          <button
            className={`top-bar-button ${clickedButtons.settings ? "clicked" : ""}`}
            onClick={() => handleStop()}
          >
            SETTINGS
          </button>
          <button
            className={`top-bar-button ${clickedButtons.practiceMode ? "clicked" : ""}`}
            onClick={() => handleButtonClick("practiceMode")}
          >
            PRACTICE MODE
          </button>
          <button
            className={`top-bar-button ${clickedButtons.toggleCamera ? "clicked" : ""}`}
            onClick={() => handleButtonClick("toggleCamera")}
          >
            TOGGLE CAMERA
          </button>
        </div>
        {clickedButtons.toggleCamera && (
          <img src="./placeholder.png" alt="Underneath" className="underneath-image" />
        )}
        {clickedButtons.practiceMode && (
          <div className="practice-log">
            <h2 className="practice-log-title">PRACTICE LOG</h2>
            <div className="practice-log-content">
              <p>Log Entry: UP AND RIGHT</p>
              <p>Log Entry: DOWN AND RIGHT</p>
              <p>Log Entry: LEFT</p>
              <p>Log Entry: UP</p>
              {/* Add dynamic entries here */}
            </div>
          </div>
        )}
      </div>
      <button className="stop-button" onClick={handleStop}>
        STOP
      </button>
    </div>
  );
};

export default InProgress;
