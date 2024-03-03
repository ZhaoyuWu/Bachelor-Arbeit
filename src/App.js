import React, { useEffect, useState, useRef } from 'react';
import './App.css';

function App() {
  const [pathData, setPathData] = useState([]); // Path data from the backend
  const [penStrokes, setPenStrokes] = useState([]); // Coordinates for storing electronic pen handwriting
  const canvasRef = useRef(null);
  const isDrawingRef = useRef(false);

  useEffect(() => {
    const fetchPathData = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/generate-path');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setPathData(data);

        console.log("Received path points count:", data.length);
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

  try {
    // Step 1: Submit handwriting data to the back office

    const submitResponse = await fetch('http://127.0.0.1:5000/submit-path', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(penStrokes),
    });

    if (!submitResponse.ok) {
      throw new Error(`HTTP error on submit! status: ${submitResponse.status}`);
    }

    // Step 2: Wait for backend processing to complete and request new path data after processing

    const getPathResponse = await fetch('http://127.0.0.1:5000/get-processed-path');
    if (!getPathResponse.ok) {
      throw new Error(`HTTP error on get processed path! status: ${getPathResponse.status}`);
    }

    const newPathData = await getPathResponse.json();
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

  pathData.forEach((point, index) => {
    const color = index === 0 ? 'green' : index === pathData.length - 1 ? 'red' : 'grey';
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(point[0], point[1], point[2] / 2, 0, 2 * Math.PI);
    ctx.fill();
  });
};

  useEffect(() => {
  if (pathData.length > 0) {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    pathData.slice(1, -1).forEach((point) => {
      ctx.fillStyle = 'grey';
      ctx.beginPath();
      ctx.arc(point[0], point[1], point[2] / 2, 0, 2 * Math.PI);
      ctx.fill();
    });

    if (pathData.length > 0) {
      const startPoint = pathData[0];
      ctx.fillStyle = 'green';
      ctx.beginPath();
      ctx.arc(startPoint[0], startPoint[1], startPoint[2] / 2, 0, 2 * Math.PI);
      ctx.fill();
    }

    if (pathData.length > 1) {
      const endPoint = pathData[pathData.length - 1];
      ctx.fillStyle = 'red';
      ctx.beginPath();
      ctx.arc(endPoint[0], endPoint[1], endPoint[2] / 2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}, [pathData]);

  return (
    <div className="App">
      <header className="App-header">
        <canvas ref={canvasRef} width={600} height={600} style={{ backgroundColor: '#FFF' }}></canvas>
        <button onClick={handleSubmit}>Submit</button>
      </header>
    </div>
  );
}

export default App;
