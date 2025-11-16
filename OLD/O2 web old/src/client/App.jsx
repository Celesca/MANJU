import { BrowserRouter, Routes, Route } from "react-router-dom";
import MicVideo from "./components/Home.jsx";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MicVideo />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
