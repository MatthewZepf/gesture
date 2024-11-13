import React, { useState } from "react";
import "./style.css";
import { useNavigate } from "react-router-dom";

export function App() {
  const keybinds = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
    "Enter", "Space", "Shift", "Control", "Alt", "Tab", "Backspace",
  ];
  const navigate = useNavigate();

  const initialPositions = {
    up: "ArrowUp",
    "up and right": "ArrowRight",
    right: "ArrowRight",
    "down and right": "ArrowDown",
    down: "ArrowDown",
    "down and left": "ArrowLeft",
    left: "ArrowLeft",
    "up and left": "ArrowUp",
  };

  const [selected, setSelected] = useState(initialPositions);

  const handleChange = (position, value) => {
    setSelected((prev) => ({
      ...prev,
      [position]: value,
    }));
  };

  const handleConfirm = () => {
    console.log("Keybinds confirmed:", selected);
    navigate("/inprogress");
  };

  return (
    <div className="App">
      <img src="../logo.png" alt="Logo" className="center-logo" />
      <div className="compass-container">
        {Object.entries(selected).map(([position, value]) => (
          <div key={position} className={`compass-box ${position.replace(/\s/g, "-")}`}>
            <div>{position.toUpperCase()}</div>
            <select style={{ width: "12vw", height: "4vh", fontSize: "2vh" }}
              value={value}
              onChange={(e) => handleChange(position, e.target.value)}
            >
              {keybinds.map((key) => (
                <option key={key} value={key}>
                  {key}
                </option>
              ))}
            </select>
          </div>
        ))}
        <button className="confirm-button" onClick={handleConfirm}>
          CONFIRM AND START
        </button>
      </div>
    </div>
  );
}

export default App;