export default function Result({value}){
  return (
    <div className="result">
      <h2>Chlorine Prediction</h2>
      <div className="value">{value} PPM</div>
    </div>
  );
}
