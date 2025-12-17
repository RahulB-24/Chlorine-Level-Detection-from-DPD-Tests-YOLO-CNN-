import {predictImage} from "../api/predict";

export default function Upload({onResult,onImage}){

  const handleChange=async(e)=>{
    const file=e.target.files[0];
    if(!file)return;

    onImage(URL.createObjectURL(file));

    try{
      const data=await predictImage(file);
      onResult(data.prediction);
    }catch(err){
      alert(err.message);
    }
  };

  return (
    <div className="upload">
      <h3>Upload Water Image</h3>
      <input type="file" accept="image/*" onChange={handleChange}/>
    </div>
  );
}
