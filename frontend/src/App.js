import {useState} from "react";
import Upload from "./components/Upload";

export default function App(){
  const[pred,setPred]=useState(0);
  const[img,setImg]=useState(null);

  return (
    <>
      <div className="water-bg"></div>

      <div className="container">
        <h1>Chlorine Detection</h1>
        <p>AI-based chlorine level estimation</p>

        <Upload onResult={setPred} onImage={setImg}/>

        {img&&<img src={img} className="preview" alt="Uploaded"/>}

        <div className="slider-container">
          <input
            type="range"
            min="0"
            max="5"
            step="0.01"
            value={pred}
            readOnly
          />
          <div className="value">{pred} PPM</div>
        </div>
      </div>
    </>
  );
}
