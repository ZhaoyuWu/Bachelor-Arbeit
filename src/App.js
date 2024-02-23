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

  drawRandomPolygons() {
    this.clearCanvas();

    const canvas = this.canvasRef.current;
    const ctx = this.ctx;
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    const numSides = Math.floor(Math.random() * 8) + 3;
    const polygon = [];
    const radius = this.cmToPixel(4);
    const centerX = canvasWidth / 2;
    const centerY = canvasHeight / 2;
    const angle = (2 * Math.PI) / numSides;

    for (let i = 0; i < numSides; i++) {
      const x = centerX + radius * Math.cos(i * angle);
      const y = centerY + radius * Math.sin(i * angle);
      polygon.push({ x, y });
    }

    this.penData.polygons.push(polygon);

    ctx.beginPath();
    ctx.moveTo(polygon[0].x, polygon[0].y);

    for (let i = 1; i < numSides; i++) {
      ctx.lineTo(polygon[i].x, polygon[i].y);
    }

    ctx.closePath();
    ctx.stroke();

    const outerNumSides = numSides * 2;
    const outerPolygon = [];
    const outerRadius = radius + this.cmToPixel(2);
    for (let i = 0; i < outerNumSides; i++) {
      const x = centerX + outerRadius * Math.cos(i * angle);
      const y = centerY + outerRadius * Math.sin(i * angle);
      outerPolygon.push({ x, y });
    }

    this.penData.outerPolygons.push(outerPolygon);

    ctx.beginPath();
    ctx.moveTo(outerPolygon[0].x, outerPolygon[0].y);

    for (let i = 1; i < outerNumSides; i++) {
      ctx.lineTo(outerPolygon[i].x, outerPolygon[i].y);
    }

    ctx.closePath();
    ctx.stroke();
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
  }

  handleTouchMove = (e) => {
    if (!this.isDrawing) return;
    const touch = e.touches[0];
    const x = touch.clientX - this.canvasRef.current.getBoundingClientRect().left;
    const y = touch.clientY - this.canvasRef.current.getBoundingClientRect().top;
    this.ctx.lineTo(x, y);
    this.ctx.stroke();

    this.penData.time.push(Date.now());
    this.penData.x.push(x);
    this.penData.y.push(y);
  }

  handleTouchEnd = () => {
    this.ctx.closePath();
    this.isDrawing = false;
  }

  handleRefreshAndSubmitData = () => {
    const distances = calculateDistancesForPenData(this.penData, this.penData.polygons[0]);
    this.penData.distances = distances;

    axios.post('http://127.0.0.1:5000/api/submit_pen_data', this.penData)
      .then(response => {
        console.log(response.data);
        this.drawRandomPolygons();
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
