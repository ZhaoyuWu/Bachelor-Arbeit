import React, { useEffect, useState, useRef } from 'react';
import './App.css';

function pixelsToCm(pixels) {
  const scaleFactor = 0.026458333;
  return pixels * scaleFactor;
}

function cmToPixels(cm) {
  const scaleFactor = 0.026458333;
  return cm / scaleFactor;
}

function App() {
  const [pathData, setPathData] = useState([]); // Path data from the backend
  const [penStrokes, setPenStrokes] = useState([]); // Coordinates for storing electronic pen handwriting
  const canvasRef = useRef(null);
  const isDrawingRef = useRef(false);
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  useEffect(() => {
  const fetchPathData = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/generate-path');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      let data = await response.json();

      data = data.map(point => [cmToPixels(point[0]), cmToPixels(point[1]), cmToPixels(point[2])]);

      setPathData(data);
      console.log("Received path points: ", data);
      console.log("Received path points count: ", data.length);
    } catch (error) {
      console.error("Could not fetch path data:", error);
    }
  };

  fetchPathData();
}, []);


  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const handleTouchStart = (event) => {
      const touch = event.touches[0];
      const x = touch.clientX - canvas.offsetLeft;
      const y = touch.clientY - canvas.offsetTop;
      isDrawingRef.current = true;
      ctx.beginPath();
      ctx.moveTo(x, y);
    };

    const handleTouchMove = (event) => {
      if (!isDrawingRef.current) return;
      const touch = event.touches[0];
      const rect = canvas.getBoundingClientRect();
      const x = touch.clientX - rect.left;
      const y = touch.clientY - rect.top;
      ctx.lineTo(x, y);
      ctx.stroke();

      setPenStrokes(prevStrokes => [...prevStrokes, [x, y]]);
    };

    const handleTouchEnd = () => {
      isDrawingRef.current = false;
    };

    canvas.addEventListener("touchstart", handleTouchStart, false);
    canvas.addEventListener("touchmove", handleTouchMove, false);
    canvas.addEventListener("touchend", handleTouchEnd, false);

    return () => {
      canvas.removeEventListener("touchstart", handleTouchStart);
      canvas.removeEventListener("touchmove", handleTouchMove);
      canvas.removeEventListener("touchend", handleTouchEnd);

      drawPath(ctx, pathData);
    };
  }, [pathData]);

  const handleSubmit = async () => {
  console.log("Submitting pen strokes count:", penStrokes.length);


  const convertedPenStrokes = penStrokes.map(stroke => [
      pixelsToCm(stroke[0] - offset.x),
      pixelsToCm(stroke[1] - offset.y),
    ]);

  try {
    const submitResponse = await fetch('http://127.0.0.1:5000/submit-path', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(convertedPenStrokes),
    });

    if (!submitResponse.ok) {
      throw new Error(`HTTP error on submit! status: ${submitResponse.status}`);
    }

    // Step 2: Wait for backend processing to complete and request new path data after processing

    const getPathResponse = await fetch('http://127.0.0.1:5000/get-processed-path');
    if (!getPathResponse.ok) {
      throw new Error(`HTTP error on get processed path! status: ${getPathResponse.status}`);
    }

    let newPathData = await getPathResponse.json();

    newPathData = newPathData.map(point => [cmToPixels(point[0]), cmToPixels(point[1]), cmToPixels(point[2])]);

    console.log("New path data received:", newPathData);

    // Step 3: Updating the canvas with new path data
    setPathData(newPathData);
    setPenStrokes([]); // clear stroke

  } catch (error) {
    console.error("Could not process path data:", error);
  }
};

const drawPath = (ctx, pathData) => {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (pathData.length === 0) return;

  const xCoords = pathData.map(point => point[0]);
  const yCoords = pathData.map(point => point[1]);
  const minX = Math.min(...xCoords);
  const maxX = Math.max(...xCoords);
  const minY = Math.min(...yCoords);
  const maxY = Math.max(...yCoords);

  const pathWidth = maxX - minX;
  const pathHeight = maxY - minY;
  const pathCenterX = minX + pathWidth / 2;
  const pathCenterY = minY + pathHeight / 2;
  const canvasCenterX = ctx.canvas.width / 2;
  const canvasCenterY = ctx.canvas.height / 2;

  const offsetX = canvasCenterX - pathCenterX;
  const offsetY = canvasCenterY - pathCenterY;

  setOffset({ x: offsetX, y: offsetY });

  pathData.slice(1, -1).forEach((point) => {
  const [x, y, radius] = point;
  ctx.fillStyle = 'grey';
  ctx.beginPath();
  ctx.arc(x + offsetX, y + offsetY, radius / 2, 0, 2 * Math.PI);
  ctx.fill();
});

// 然后绘制起点，确保它在其他点之上
if (pathData.length > 0) {
  const [startX, startY, startRadius] = pathData[0];
  ctx.fillStyle = 'green';
  ctx.beginPath();
  ctx.arc(startX + offsetX, startY + offsetY, startRadius / 2, 0, 2 * Math.PI);
  ctx.fill();
}

// 最后绘制终点
if (pathData.length > 1) {
  const [endX, endY, endRadius] = pathData[pathData.length - 1];
  ctx.fillStyle = 'red';
  ctx.beginPath();
  ctx.arc(endX + offsetX, endY + offsetY, endRadius / 2, 0, 2 * Math.PI);
  ctx.fill();
}
};


  useEffect(() => {
  if (pathData.length > 0 && canvasRef.current) {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    drawPath(ctx, pathData);
  }
}, [pathData]);

  return (
    <div className="App">
      <header className="App-header">
        <canvas ref={canvasRef} width={800} height={800} style={{ backgroundColor: '#FFF' }}></canvas>
        <button onClick={handleSubmit}>Submit</button>
      </header>
    </div>
  );
}

export default App;
