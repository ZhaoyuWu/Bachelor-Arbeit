import React, { useEffect, useState, useRef } from "react";
import { View, Button, StyleSheet } from "react-native";
import Svg, { Path, Circle } from "react-native-svg";

const scaleFactor = 0.05;

function pixelsToCm(pixels) {
  return pixels * scaleFactor;
}

function cmToPixels(cm) {
  return cm / scaleFactor;
}

export default function App() {
  const [pathData, setPathData] = useState([]);
  const [penStrokes, setPenStrokes] = useState([]);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const isDrawingRef = useRef(false);

  useEffect(() => {
    const fetchPathData = async () => {
      try {
        const response = await fetch("http://192.168.2.210:5000/generate-path");
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        let data = await response.json();
        data = data.map((point) => [cmToPixels(point[0]), cmToPixels(point[1]), cmToPixels(point[2])]);

        setPathData(data);
      } catch (error) {
        console.error("Could not fetch path data:", error);
      }
    };

    fetchPathData();
  }, []);
  const calculateBoundingBox = (data) => {
    const minX = Math.min(...data.map((point) => point[0]));
    const maxX = Math.max(...data.map((point) => point[0]));
    const minY = Math.min(...data.map((point) => point[1]));
    const maxY = Math.max(...data.map((point) => point[1]));

    const width = maxX - minX;
    const height = maxY - minY;

    const centerX = minX + width / 2;
    const centerY = minY + height / 2;

    return { width, height, centerX, centerY };
  };

  const handleTouchStart = (event) => {
    const { locationX, locationY } = event.nativeEvent;
    isDrawingRef.current = true;
    setPenStrokes([[locationX, locationY]]);
  };

  const handleTouchMove = (event) => {
    if (!isDrawingRef.current) return;
    const { locationX, locationY } = event.nativeEvent;
    setPenStrokes((prevStrokes) => [...prevStrokes, [locationX, locationY]]);
  };

  const handleTouchEnd = () => {
    isDrawingRef.current = false;
  };

  const handleSubmit = async () => {
    console.log("Submitting pen strokes count:", penStrokes.length);

    const convertedPenStrokes = penStrokes.map((stroke) => [
      pixelsToCm(stroke[0] - offset.x),
      pixelsToCm(stroke[1] - offset.y),
    ]);

    try {
      const submitResponse = await fetch("http://192.168.2.210:5000/submit-path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(convertedPenStrokes),
      });

      if (!submitResponse.ok) throw new Error(`HTTP error on submit! status: ${submitResponse.status}`);

      const getPathResponse = await fetch("http://192.168.2.210:5000/get-processed-path");
      if (!getPathResponse.ok) throw new Error(`HTTP error on get processed path! status: ${getPathResponse.status}`);

      let newPathData = await getPathResponse.json();
      newPathData = newPathData.map((point) => [cmToPixels(point[0]), cmToPixels(point[1]), cmToPixels(point[2])]);

      setPathData(newPathData);
      setPenStrokes([]);
    } catch (error) {
      console.error("Could not process path data:", error);
    }
  };

  const renderPath = () => {
    if (pathData.length === 0) return null;

    const { width, height, centerX, centerY } = calculateBoundingBox(pathData);

    const offsetX = (width < 400 ? (400 - width) / 2 : 0);
    const offsetY = (height < 280 ? (280 - height) / 2 : 0);

    return (
      <>
        {pathData.slice(1, -1).map(([x, y, radius], index) => (
          <Circle key={index} cx={x + offsetX} cy={y + offsetY} r={radius / 2} fill="grey" />
        ))}

        {pathData.length > 0 && (
          <Circle cx={pathData[0][0] + offsetX} cy={pathData[0][1] + offsetY} r={pathData[0][2] / 2} fill="green" />
        )}

        {pathData.length > 1 && (
          <Circle
            cx={pathData[pathData.length - 1][0] + offsetX}
            cy={pathData[pathData.length - 1][1] + offsetY}
            r={pathData[pathData.length - 1][2] / 2}
            fill="red"
          />
        )}

        </>
    );
  };

  const renderPenStrokes = () => {
    if (penStrokes.length === 0) return null;

    let strokePath = `M${penStrokes[0][0]},${penStrokes[0][1]}`;
    for (let i = 1; i < penStrokes.length; i++) {
      strokePath += ` L${penStrokes[i][0]},${penStrokes[i][1]}`;
    }

    return <Path d={strokePath} stroke="blue" strokeWidth="2" fill="none" />;
  };

  return (
    <View style={styles.container}>
      <View
        style={styles.drawingArea}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        <Svg width="100%" height="100%" viewBox="0 0 250 250">
          {renderPath()}
          {renderPenStrokes()}
        </Svg>
      </View>
      <Button title="Submit" onPress={handleSubmit} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#FFF",
  },
  drawingArea: {
    width: 300,
    height: 300,
    borderWidth: 1,
    borderColor: "#000",
    backgroundColor: "#FFF",
  },
});
