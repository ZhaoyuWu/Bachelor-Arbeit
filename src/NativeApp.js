import React, { useEffect, useState, useRef } from "react";
import { View, Button, StyleSheet } from "react-native";
import Svg, { Path, Circle } from "react-native-svg";

const scaleFactor = 0.026458333;

function pixelsToCm(pixels) {
  return pixels * scaleFactor;
}

function cmToPixels(cm) {
  return cm / scaleFactor;
}

export default function App() {
  const [pathData, setPathData] = useState([]); // 后端返回的路径数据
  const [penStrokes, setPenStrokes] = useState([]); // 电子笔记录的手写数据
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const isDrawingRef = useRef(false);

  // 获取路径数据
  useEffect(() => {
    const fetchPathData = async () => {
      try {
        const response = await fetch("http://localhost:5000/generate-path");
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

  // 处理触摸事件
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

  // 提交数据
  const handleSubmit = async () => {
    console.log("Submitting pen strokes count:", penStrokes.length);

    const convertedPenStrokes = penStrokes.map((stroke) => [
      pixelsToCm(stroke[0] - offset.x),
      pixelsToCm(stroke[1] - offset.y),
    ]);

    try {
      const submitResponse = await fetch("http://127.0.0.1:5000/submit-path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(convertedPenStrokes),
      });

      if (!submitResponse.ok) throw new Error(`HTTP error on submit! status: ${submitResponse.status}`);

      const getPathResponse = await fetch("http://127.0.0.1:5000/get-processed-path");
      if (!getPathResponse.ok) throw new Error(`HTTP error on get processed path! status: ${getPathResponse.status}`);

      let newPathData = await getPathResponse.json();
      newPathData = newPathData.map((point) => [cmToPixels(point[0]), cmToPixels(point[1]), cmToPixels(point[2])]);

      setPathData(newPathData);
      setPenStrokes([]); // 清空手写轨迹
    } catch (error) {
      console.error("Could not process path data:", error);
    }
  };

  // 生成 SVG 路径
  const renderPath = () => {
    if (pathData.length === 0) return null;

    let pathString = `M${pathData[0][0]},${pathData[0][1]}`;
    for (let i = 1; i < pathData.length; i++) {
      pathString += ` L${pathData[i][0]},${pathData[i][1]}`;
    }

    return (
      <>
        {/* 灰色路径点 */}
        {pathData.slice(1, -1).map(([x, y, radius], index) => (
          <Circle key={index} cx={x + offset.x} cy={y + offset.y} r={radius / 2} fill="grey" />
        ))}

        {/* 绿色起点 */}
        {pathData.length > 0 && (
          <Circle cx={pathData[0][0] + offset.x} cy={pathData[0][1] + offset.y} r={pathData[0][2] / 2} fill="green" />
        )}

        {/* 红色终点 */}
        {pathData.length > 1 && (
          <Circle
            cx={pathData[pathData.length - 1][0] + offset.x}
            cy={pathData[pathData.length - 1][1] + offset.y}
            r={pathData[pathData.length - 1][2] / 2}
            fill="red"
          />
        )}

        {/* 绘制路径 */}
        <Path d={pathString} stroke="black" strokeWidth="2" fill="none" />
      </>
    );
  };

  // 生成手写轨迹
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
        <Svg width="100%" height="100%">
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
