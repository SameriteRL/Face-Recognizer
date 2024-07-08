"use client";

import { useState } from "react";
import axios from "axios";

const errorMessage = () => {
  return (
    <>
      <p className="mb-5">An error occurred while processing your images!</p>
      <ul>
        <li>1. Make sure you've uploaded two image files.</li>
        <li>2. Ensure at least one face is visible and as clear as possible in each image.</li>
      </ul>
    </>
  )
}

export default function Home() {
  const [faceFile, setFaceFile] = useState(null);
  const [testFile, setTestFile] = useState(null);
  const [showError, setShowError] = useState(false);
  const [imageSrc, setImageSrc] = useState("");
  const [formDisabled, setFormDisabled] = useState(false);

  const handleFaceFileChange = (e) => {
    setFaceFile(e.target.files[0]);
  };

  const handleTestFileChange = (e) => {
    setTestFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("faceImg", faceFile);
    formData.append("testImg", testFile);
    setFormDisabled(true);
    try {
      const response = await axios.post(
        "http://localhost:8080/submit",
        formData,
        {
          headers: {"Content-Type": "multipart/form-data"},
          responseType: "json"
        }
      );
      console.log("Files uploaded");
      try {
        const base64img = response.data.result;
        const imgUrl = `data:image/jpeg;base64,${base64img}`;
        setImageSrc(imgUrl);
        console.log("Image rendered")
        setShowError(false);
      }
      catch (error) {
        console.log("Error rendering image", error);
        setImageSrc("");
        setShowError(true);
      }
    }
    catch (error) {
      console.error("Error uploading file", error);
      setImageSrc("");
      setShowError(true);
    }
    setFormDisabled(false);
  };

  return (
    <div className="flex flex-col items-center justify-center">
      <form className="flex flex-col items-center overflow-auto box-border p-8">
        <label className="text-2xl">Upload a face less than 10MB</label>
        <input className="mb-7" type="file" accept="image/*" onChange={handleFaceFileChange} disabled={formDisabled} />
        <label className="text-2xl">Upload a test image less than 10MB</label>
        <input className="mb-7" type="file" accept="image/*" onChange={handleTestFileChange} disabled={formDisabled} />
        <button className="bg-red-700 text-white font-bold mb-4 py-2 px-4 rounded hover:bg-red-800" onClick={handleSubmit}>Submit</button>
      </form>
      {
        imageSrc ? (
          <img src={imageSrc} width={500} height={500} />
        ) : (
          showError ? errorMessage() : <p>Waiting for output...</p>)
      }
    </div>
  );
}
