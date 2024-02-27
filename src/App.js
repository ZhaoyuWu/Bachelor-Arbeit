import React, { useEffect, useState, useRef } from 'react';
import './App.css';

function App() {
  const [pathData, setPathData] = useState([]);
  const canvasRef = useRef(null);

  useEffect(() => {
    const fetchPathData = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/generate-path');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setPathData(data);
      } catch (error) {
        console.error("Could not fetch path data:", error);
      }
    };

    fetchPathData();
  }, []);

  useEffect(() => {
    if (pathData.length > 0) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      const xValues = pathData.map(point => point[0]);
      const yValues = pathData.map(point => point[1]);
      const minX = Math.min(...xValues);
      const maxX = Math.max(...xValues);
      const minY = Math.min(...yValues);
      const maxY = Math.max(...yValues);
      const pathWidth = maxX - minX;
      const pathHeight = maxY - minY;
      const scaleX = width / (pathWidth || 1);
      const scaleY = height / (pathHeight || 1);
      const scale = Math.min(scaleX, scaleY) * 0.9;
      const offsetX = (width - pathWidth * scale) / 2 - minX * scale;
      const offsetY = (height - pathHeight * scale) / 2 - minY * scale;

      pathData.slice(1, -1).forEach(point => {
        drawCircle(ctx, point, scale, offsetX, offsetY, 'grey');
      });

      if (pathData.length > 0) {
        drawCircle(ctx, pathData[0], scale, offsetX, offsetY, 'green');
      }

      if (pathData.length > 1) {
        drawCircle(ctx, pathData[pathData.length - 1], scale, offsetX, offsetY, 'red');
      }
    }
  }, [pathData]);

  const drawCircle = (ctx, point, scale, offsetX, offsetY, color) => {
    const x = point[0] * scale + offsetX;
    const y = point[1] * scale + offsetY;
    const diameter = point[2]/2 * scale;

    ctx.beginPath();
    ctx.arc(x, y, diameter / 2, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  };

  return (
    <div className="App">
      <header className="App-header">
        <canvas ref={canvasRef} width={800} height={800} style={{ backgroundColor: '#FFF' }}></canvas>
      </header>
    </div>
  );
}

export default App;
