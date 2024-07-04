"use client";

import { useState } from "react";
import axios from "axios";

export default function Home() {

  const [faceFile, setFaceFile] = useState(null);
  const [testFile, setTestFile] = useState(null);
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
          responseType: "blob"
        }
      );
      console.log("Files uploaded");
      try {
        const url = URL.createObjectURL(response.data);
        setImageSrc(url);
        console.log("Image rendered")
      }
      catch (error) {
        console.log("Error rendering image", error);
      }
    }
    catch (error) {
      console.error("Error uploading file", error);
    }
    setFormDisabled(false);
  };

  return (
    <div className="flex flex-col items-center justify-center">
      <form className="flex flex-col items-center overflow-auto box-border p-8">
        <label className="text-2xl">Upload a face less than 10MB</label>
        <input className="mb-7" type="file" accept="image/*" onChange={handleFaceFileChange} disabled={formDisabled} />
        <label className="text-2xl">Upload a test image less than 10MB</label>
        <input className="mb-2" type="file" accept="image/*" onChange={handleTestFileChange} disabled={formDisabled} />
        <button className="bg-red-700 text-white font-bold mt-5 py-2 px-4 rounded hover:bg-red-800" onClick={handleSubmit}>Submit</button>
      </form>
      {
        imageSrc ?
          <img src={imageSrc} width={500} height={500} />
          :
          <p>Waiting for output...</p>
      }
    </div>
  );
}
