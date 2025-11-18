import { BrowserRouter, Routes, Route } from "react-router-dom";
import MicVideo from "./components/Home";
import Navbar from "./components/Navbar";
import About from "./components/About";
import * as LoginModule from "./components/Login";
import * as SignUpModule from "./components/SignUp";

// Resolve either default export or named export (defensive for mixed module styles)
const Login = LoginModule.default ?? LoginModule.Login ?? (() => null);
const SignUp = SignUpModule.default ?? SignUpModule.SignUp ?? (() => null);
import { useState } from "react";


function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  return (
    <BrowserRouter>
      <Navbar isLoggedIn={isLoggedIn} setIsLoggedIn={setIsLoggedIn} />
      <Routes>
        <Route path="/" element={<MicVideo />} />
        <Route path="/about" element={<About />} />
        <Route path="/login" element={<Login setIsLoggedIn={setIsLoggedIn} />} />
        <Route path="/signup" element={<SignUp setIsLoggedIn={setIsLoggedIn} />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
