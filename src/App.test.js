import React, { Component } from 'react';
import axios from 'axios';

function calculateDistance(point, polygon) {
  function pointToSegmentDistance(p, p1, p2) {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const t = ((p.x - p1.x) * dx + (p.y - p1.y) * dy) / (dx * dx + dy * dy);

    if (t < 0) {
      return Math.sqrt((p.x - p1.x) ** 2 + (p.y - p1.y) ** 2);
    }

    if (t > 1) {
      return Math.sqrt((p.x - p2.x) ** 2 + (p.y - p2.y) ** 2);
    }

    const closestX = p1.x + t * dx;
    const closestY = p1.y + t * dy;

    return Math.sqrt((p.x - closestX) ** 2 + (p.y - closestY) ** 2);
  }

  const distances = [];
  for (let i = 0; i < polygon.length; i++) {
    const p1 = polygon[i];
    const p2 = polygon[(i + 1) % polygon.length];
    const dist = pointToSegmentDistance(point, p1, p2);
    distances.push(dist);
  }
  return Math.min(...distances);
}

function calculateDistancesForPenData(penData, polygon) {
  const distances = [];
  for (let i = 0; i < penData.time.length; i++) {
    const point = { x: penData.x[i], y: penData.y[i] };
    const distance = calculateDistance(point, polygon);
    distances.push({ distance });
  }
  return distances;
}

class DrawingApp extends Component {
  constructor() {
    super();
    this.canvasRef = React.createRef();
    this.ctx = null;
    this.isDrawing = false;
    this.polygonsGenerated = false;
    this.numHearts = 3;
    this.score = 0;
    this.penData = {
      time: [],
      x: [],
      y: [],
      polygons: [],
      outerPolygons: [],
      distances: [],
    };
  }

  componentDidMount() {
    this.ctx = this.canvasRef.current.getContext('2d');
    this.canvasRef.current.addEventListener('touchstart', this.handleTouchStart);
    this.canvasRef.current.addEventListener('touchmove', this.handleTouchMove);
    this.canvasRef.current.addEventListener('touchend', this.handleTouchEnd);

    if (!this.polygonsGenerated) {
      this.drawRandomPolygons();
      this.polygonsGenerated = true;
    }

    this.updateScoreDisplay();
  }

  cmToPixel(cm) {
    const dpi = 96;
    return (cm * dpi) / 2.54;
  }

  clearCanvas() {
    const canvas = this.canvasRef.current;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  updateScoreDisplay() {

    this.ctx.fillStyle = 'lightgrey';
    this.ctx.fillRect(0, 0, 100, 20);


    this.ctx.font = '14px Arial';
    this.ctx.fillStyle = 'black';


    this.ctx.fillText(`Score: ${this.score}`, 10, 15);
  }

  drawRandomPolygons() {
    this.clearCanvas();
    this.ctx.fillStyle = 'lightgrey';
    this.ctx.fillRect(0, 0, this.canvasRef.current.width, this.canvasRef.current.height);

    const canvas = this.canvasRef.current;
    const ctx = this.ctx;
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;


    const outerNumSides = Math.floor(Math.random() * 8) + 3;
    const outerPolygon = [];
    const outerRadius = this.cmToPixel(7);
    const centerX = canvasWidth / 2;
    const centerY = canvasHeight / 2;
    const angle = (2 * Math.PI) / outerNumSides;

    for (let i = 0; i < outerNumSides; i++) {
      const x = centerX + outerRadius * Math.cos(i * angle);
      const y = centerY + outerRadius * Math.sin(i * angle);
      outerPolygon.push({ x, y });
    }

    this.penData.outerPolygons = [];
    this.penData.outerPolygons.push(outerPolygon);

    ctx.beginPath();
    ctx.moveTo(outerPolygon[0].x, outerPolygon[0].y);

    for (let i = 1; i < outerNumSides; i++) {
      ctx.lineTo(outerPolygon[i].x, outerPolygon[i].y);
    }

    ctx.closePath();
    ctx.stroke();

    ctx.fillStyle = 'white';
    ctx.fill();


    const innerNumSides = outerNumSides;
    const innerPolygon = [];
    const innerRadius = this.cmToPixel(4);
    for (let i = 0; i < innerNumSides; i++) {
      const x = centerX + innerRadius * Math.cos(i * angle);
      const y = centerY + innerRadius * Math.sin(i * angle);
      innerPolygon.push({ x, y });
    }

    this.penData.polygons = [];
    this.penData.polygons.push(innerPolygon);

    ctx.beginPath();
    ctx.moveTo(innerPolygon[0].x, innerPolygon[0].y);

    for (let i = 1; i < innerNumSides; i++) {
      ctx.lineTo(innerPolygon[i].x, innerPolygon[i].y);
    }

    ctx.closePath();
    ctx.stroke();

    ctx.fillStyle = 'lightgrey';
    ctx.fill();

    const randomvertex = Math.floor(Math.random() * outerPolygon.length)

    const randomVertex1 = outerPolygon[randomvertex];
    const randomVertex2 = innerPolygon[randomvertex];


  const midPointX = (randomVertex1.x + randomVertex2.x) / 2;
  const midPointY = (randomVertex1.y + randomVertex2.y) / 2;


  ctx.beginPath();
  ctx.arc(midPointX, midPointY, this.cmToPixel(0.7), 0, 2 * Math.PI);
  ctx.fillStyle = 'red';
  ctx.fill();
  ctx.closePath();

  console.log(this.penData.polygons);

  this.recordCanvasState();

  for (let i = 0; i < this.numHearts; i++) {
      this.drawHeart(canvasWidth - 20 * (i + 1), 20);
    }

  }

  drawHeart(x, y) {
    this.ctx.fillStyle = 'red';
    this.ctx.beginPath();
    this.ctx.moveTo(x, y + 5);
    this.ctx.bezierCurveTo(x + 5, y, x + 10, y, x + 10, y + 12.5);
    this.ctx.bezierCurveTo(x + 10, y + 15, x + 5, y + 22, x, y + 30);
    this.ctx.bezierCurveTo(x - 5, y + 22, x - 10, y + 15, x - 10, y + 12.5);
    this.ctx.bezierCurveTo(x - 10, y, x - 5, y, x, y + 5);
    this.ctx.closePath();
    this.ctx.fill();
  }

  recordCanvasState() {
    this.recordedImageData = this.ctx.getImageData(
      0,
      0,
      this.canvasRef.current.width,
      this.canvasRef.current.height
    );
  }

  handleTouchStart = (e) => {
    this.ctx.beginPath();
    const touch = e.touches[0];
    const x = touch.clientX - this.canvasRef.current.getBoundingClientRect().left;
    const y = touch.clientY - this.canvasRef.current.getBoundingClientRect().top;
    this.ctx.moveTo(x, y);

    this.isDrawing = true;
    this.penData.time.push(Date.now());
    this.penData.x.push(x);
    this.penData.y.push(y);

    if (this.isInRedArea(x, y)) {

    }else
    {
    this.isDrawing = false;
      alert("Please start from red area!");

      this.penData.time = [];
      this.penData.x = [];
      this.penData.y = [];
      this.penData.distances = [];

      this.ctx.putImageData(this.recordedImageData, 0, 0);
      this.updateScoreDisplay();
      for (let i = 0; i < this.numHearts; i++) {
      this.drawHeart(this.canvasRef.current.width - 20 * (i + 1), 20);
    }
    }
  }

  handleTouchMove = (e) => {
    if (!this.isDrawing) return;

    const touch = e.touches[0];
    const x = touch.clientX - this.canvasRef.current.getBoundingClientRect().left;
    const y = touch.clientY - this.canvasRef.current.getBoundingClientRect().top;

    if (!this.isInGrayArea(x, y)) {
      this.ctx.lineTo(x, y);
      this.ctx.stroke();

      this.penData.time.push(Date.now());
      this.penData.x.push(x);
      this.penData.y.push(y);
    } else {
      this.isDrawing = false;


      if(this.numHearts === 0)
      {
      alert("Pen outside white area!");
      alert("Game Over!");
      this.drawRandomPolygons();
      this.numHearts = 3;
      for (let i = 0; i < this.numHearts; i++) {
      this.drawHeart(this.canvasRef.current.width - 20 * (i + 1), 20);
    }
      }else{
      alert("Pen outside white area!");

      this.penData.time = [];
      this.penData.x = [];
      this.penData.y = [];
      this.penData.distances = [];

        this.ctx.putImageData(this.recordedImageData, 0, 0);
      this.updateScoreDisplay();
      this.numHearts--;
      for (let i = 0; i < this.numHearts; i++) {
      this.drawHeart(this.canvasRef.current.width - 20 * (i + 1), 20);
    }
      }
    }
  }

  handleTouchEnd = () => {
  const touchEndX = this.penData.x[this.penData.x.length - 1];
  const touchEndY = this.penData.y[this.penData.y.length - 1];

  if (this.isInRedArea(touchEndX, touchEndY)) {
  this.ctx.closePath();
  this.isDrawing = false;
  this.score += 1;

  console.log(this.penData.polygons);

  } else {
    if(this.isInWhiteArea(touchEndX, touchEndY)){
    if(!this.isDrawing){return;}
    alert("Please end in red area!");}
    this.ctx.putImageData(this.recordedImageData, 0, 0);

      this.penData.time = [];
      this.penData.x = [];
      this.penData.y = [];
      this.penData.distances = [];

    for (let i = 0; i < this.numHearts; i++) {
      this.drawHeart(this.canvasRef.current.width - 20 * (i + 1), 20);
    }
    this.isDrawing = false;
    console.log(this.penData.polygons);
  }
  this.updateScoreDisplay();
}

  isInWhiteArea(x, y) {
    const pixel = this.ctx.getImageData(x, y, 1, 1).data;
    return pixel[0] === 255 && pixel[1] === 255 && pixel[2] === 255;
  }

  isInRedArea(x, y) {
  const pixel = this.ctx.getImageData(x, y, 1, 1).data;
  return pixel[0] === 255 && pixel[1] === 0 && pixel[2] === 0;
}

isInGrayArea(x, y) {
  const pixel = this.ctx.getImageData(x, y, 1, 1).data;
  return pixel[0] === 211 && pixel[1] === 211 && pixel[2] === 211;
}

  handleRefreshAndSubmitData = () => {
    const distances = calculateDistancesForPenData(this.penData, this.penData.polygons[0]);
    this.penData.distances = distances;

    console.log(this.penData.polygons.length);

    if (this.penData.polygons.length === 0) {
    console.error('Polygons array is empty or undefined.');
    return;
  }
    axios.post('http://127.0.0.1:5000/api/submit_pen_data', this.penData)
      .then(response => {
        console.log(response.data);
        this.drawRandomPolygons();
        this.updateScoreDisplay();
      })
      .catch(error => {
        console.error(error);
      });
  }

  render() {
    return (
      <div>
        <canvas
          ref={this.canvasRef}
          width={800}
          height={600}
        />
        <button onClick={this.handleRefreshAndSubmitData}>Next</button>
      </div>
    );
  }
}

export default DrawingApp;
