import React, { useRef, useState, useEffect } from "react";

import folkSound from "../assets/FOLK1.wav";
import kaewSound from "../assets/Keaw.wav";
import otwoSound from "../assets/hi.mp3";
import pic from "../assets/1080-6bw-removebg-preview.png";
import globe from "../assets/Generated Image September 14, 2025 - 1_50PM.png"; // ใช้ภาพลูกบอลที่คุณมีใน assets

export default function MicVideo() {
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0);

  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const dataArrayRef = useRef(null);
  const animationIdRef = useRef(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // สร้าง AudioContext และ Source แค่ครั้งเดียว
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    // เก็บความยาวไฟล์
    const onLoadedMetadata = () => {
      setDuration(audio.duration);
    };

    // อัปเดตเวลาเล่นทุก frame
    const onTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    audio.addEventListener("loadedmetadata", onLoadedMetadata);
    audio.addEventListener("timeupdate", onTimeUpdate);

    return () => {
      audio.removeEventListener("loadedmetadata", onLoadedMetadata);
      audio.removeEventListener("timeupdate", onTimeUpdate);
    };
  }, []);

  useEffect(() => {
    if (!audioContextRef.current && audioRef.current) {
      audioContextRef.current = new (window.AudioContext ||
        window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      dataArrayRef.current = new Uint8Array(analyserRef.current.fftSize);

      sourceRef.current = audioContextRef.current.createMediaElementSource(
        audioRef.current
      );
      sourceRef.current.connect(analyserRef.current);
      analyserRef.current.connect(audioContextRef.current.destination);
    }
  }, []);

  // Volume Animation
  useEffect(() => {
    const tick = () => {
      if (!analyserRef.current) return;
      analyserRef.current.getByteTimeDomainData(dataArrayRef.current);
      const sum = dataArrayRef.current.reduce(
        (a, b) => a + Math.abs(b - 128),
        0
      );
      const avg = sum / dataArrayRef.current.length;
      setVolume(avg / 20);
      animationIdRef.current = requestAnimationFrame(tick);
    };

    if (isPlaying) {
      if (audioContextRef.current.state === "suspended") {
        audioContextRef.current.resume();
      }
      tick();
    } else {
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
      setVolume(0);
    }

    return () => {
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
    };
  }, [isPlaying]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) audioRef.current.pause();
    else audioRef.current.play();
    setIsPlaying(!isPlaying);
  };

  const [audioSrc, setAudioSrc] = useState(folkSound);

  const handleSelect = (e) => {
    const value = e.target.value;
    if (value === "Folk") setAudioSrc(folkSound);
    else if (value === "Kaew") setAudioSrc(kaewSound);
    else if (value === "Otwo") setAudioSrc(otwoSound);
  };

  return (
    <div className="min-h-screen bg-black text-white flex flex-col items-center px-6 py-12">
      <div className="max-w-6xl w-full grid md:grid-cols-2 gap-12 items-center">
        {/* Left Section */}
        <div>
          <h1 className="text-5xl font-bold leading-snug">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">
              AI Multi-agent
            </span>
            <br />
            Call Center
          </h1>
          <p className="mt-6 text-lg text-gray-300">
            High-performance Thai ASR (Automatic Speech Recognition)
          </p>

          {/* Divider */}
          <div className="mt-6 h-[1px] w-48 bg-gradient-to-r from-purple-600 to-cyan-400"></div>

          {/* Button */}
          <button className="mt-8 inline-flex items-center space-x-2 border border-purple-500 text-white px-6 py-3 rounded-full hover:bg-purple-600 transition">
            <span>Get Started</span>
            <span>→</span>
          </button>
        </div>

        {/* Right Section */}
        <div className="flex justify-center">
          <img
            src={globe}
            alt="AI Globe"
            className="w-80 md:w-[400px] drop-shadow-[0_0_30px_rgba(168,85,247,0.8)]"
          />
        </div>
      </div>

      {/* Divider */}
      <div className="w-full max-w-6xl border-t border-purple-600 my-12"></div>

      <div className="w-full max-w-6xl grid md:grid-cols-2 gap-12">
        {/* Left Section */}
        <div className="flex flex-col justify-center">
          <h1 className="text-4xl font-bold">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">
              CONVERT TEXT
            </span>{" "}
            TO SPEECH
          </h1>
          <p className="mt-6 text-gray-300 leading-relaxed">
            Easily convert text into realistic speech. Supports Thai language,
            helping you work smoothly whether it’s for voiceovers, dubbing,
            teaching materials, news reading, presentation slides, podcasts, or
            reading novels. Finish your work easily, anytime, anywhere.
          </p>

          <button className="mt-8 border border-purple-400 text-purple-400 px-6 py-3 rounded-full hover:bg-purple-600 hover:text-white transition">
            LET&apos;S TRY FREE
          </button>
        </div>

        {/* Right Section */}
        <div className="bg-neutral-800 rounded-2xl p-6 shadow-lg">
          {/* Language Toggle */}
          <div className="flex space-x-2 mb-4">
            <button className="px-4 py-2 rounded-lg bg-neutral-700 flex items-center space-x-2">
              <span>🇹🇭</span>
              <span>Thai</span>
            </button>
            <button className="px-4 py-2 rounded-lg bg-neutral-700 flex items-center space-x-2">
              <span>EH</span>
              <span>English</span>
            </button>
          </div>

          {/* Dropdown + Textarea */}
          <select
            className="w-full p-3 rounded-lg bg-neutral-700 text-white mb-4"
            onChange={handleSelect}
            value={
              audioSrc === folkSound
                ? "Folk"
                : audioSrc === kaewSound
                ? "Kaew"
                : "Otwo"
            }
          >
            <option>Folk</option>
            <option>Kaew</option>
            <option>Otwo</option>
          </select>
          <textarea
            className="w-full p-3 rounded-lg bg-neutral-700 text-white h-32"
            placeholder="ลองพิมพ์ประโยคของคุณต้องการที่นี่"
          ></textarea>

          <button className="mt-4 bg-black px-6 py-2 rounded-lg hover:bg-purple-600 transition">
            Generate
          </button>
        </div>
      </div>

      {/* Divider */}
      <div className="w-full max-w-6xl border-t border-purple-600 my-12"></div>

      <div className="flex flex-col items-center space-y-6">
        {/* Mic Button */}
        <div
          onClick={togglePlay}
          className="relative flex items-center justify-center cursor-pointer"
        >
          {/* วงกลมใหญ่ ขยายตาม volume */}
          <div
            className="w-40 h-40 rounded-full bg-gradient-to-r from-purple-600 to-pink-500 shadow-lg transition-transform duration-100"
            style={{ transform: `scale(${1 + volume / 2})` }}
          ></div>

          {/* วงกลมเล็กตรงกลาง */}
          <div
            className="absolute w-20 h-20 rounded-full bg-gradient-to-r from-pink-500 to-purple-600 items-center transition-transform duration-100 flex justify-center"
            style={{ transform: `scale(${1 + volume / 2})` }}
          >
            <span className="text-4xl">🎤</span>
          </div>
        </div>

        {/* Audio Player */}
        <div className="mt-6 flex items-center space-x-2 w-64 bg-neutral-800 px-3 py-2 rounded-full">
          <button onClick={togglePlay} className="text-gray-400">
            {isPlaying ? "⏸" : "▶"}
          </button>
          <div className="flex-1 h-1 bg-gray-600 rounded">
            <div
              className="h-1 bg-purple-500 rounded"
              style={{ width: `${(currentTime / duration) * 100}%` }}
            ></div>
          </div>

          {/* เวลาแสดงเป็นนาที:วินาที */}
          <span className="text-sm">
            {Math.floor(currentTime / 60)}:
            {("0" + Math.floor(currentTime % 60)).slice(-2)}
          </span>
        </div>

         <audio ref={audioRef} src={audioSrc} />
      </div>

      {/* Divider */}
      <div className="border-t my-12"></div>

      <div className="grid grid-cols-3 gap-x-8 gap-y-4">
        <div>
          <div className="card bg-neutral-800 w-96 shadow-sm">
            <figure className="px-10 pt-10">
              <img
                src="https://img.daisyui.com/images/stock/photo-1606107557195-0e29a4b5b4aa.webp"
                alt="Shoes"
                className="rounded-xl"
              />
            </figure>
            <div className="card-body items-center text-center">
              <h2 className="card-title">Card Title</h2>
              <p>
                A card component has a figure, a body part, and inside body
                there are title and actions parts
              </p>
              <div className="card-actions">
                <button className="btn btn-primary">Buy Now</button>
              </div>
            </div>
          </div>
        </div>
        <div>
          <div className="card bg-neutral-800 w-96 shadow-sm">
            <figure className="px-10 pt-10">
              <img
                src="https://img.daisyui.com/images/stock/photo-1606107557195-0e29a4b5b4aa.webp"
                alt="Shoes"
                className="rounded-xl"
              />
            </figure>
            <div className="card-body items-center text-center">
              <h2 className="card-title">Card Title</h2>
              <p>
                A card component has a figure, a body part, and inside body
                there are title and actions parts
              </p>
              <div className="card-actions">
                <button className="btn btn-primary">Buy Now</button>
              </div>
            </div>
          </div>
        </div>
        <div>
          <div className="card bg-neutral-800 w-96 shadow-sm">
            <figure className="px-10 pt-10">
              <img
                src="https://img.daisyui.com/images/stock/photo-1606107557195-0e29a4b5b4aa.webp"
                alt="Shoes"
                className="rounded-xl"
              />
            </figure>
            <div className="card-body items-center text-center">
              <h2 className="card-title">Card Title</h2>
              <p>
                A card component has a figure, a body part, and inside body
                there are title and actions parts
              </p>
              <div className="card-actions">
                <button className="btn btn-primary">Buy Now</button>
              </div>
            </div>
          </div>
        </div>
        <div>
          <div className="card bg-neutral-800 w-96 shadow-sm">
            <figure className="px-10 pt-10">
              <img
                src="https://img.daisyui.com/images/stock/photo-1606107557195-0e29a4b5b4aa.webp"
                alt="Shoes"
                className="rounded-xl"
              />
            </figure>
            <div className="card-body items-center text-center">
              <h2 className="card-title">Card Title</h2>
              <p>
                A card component has a figure, a body part, and inside body
                there are title and actions parts
              </p>
              <div className="card-actions">
                <button className="btn btn-primary">Buy Now</button>
              </div>
            </div>
          </div>
        </div>
        <div>
          <div className="card bg-neutral-800 w-96 shadow-sm">
            <figure className="px-10 pt-10">
              <img
                src="https://img.daisyui.com/images/stock/photo-1606107557195-0e29a4b5b4aa.webp"
                alt="Shoes"
                className="rounded-xl"
              />
            </figure>
            <div className="card-body items-center text-center">
              <h2 className="card-title">Card Title</h2>
              <p>
                A card component has a figure, a body part, and inside body
                there are title and actions parts
              </p>
              <div className="card-actions">
                <button className="btn btn-primary">Buy Now</button>
              </div>
            </div>
          </div>
        </div>
        <div>
          <div className="card bg-neutral-800 w-96 shadow-sm">
            <figure className="px-10 pt-10">
              <img
                src="https://img.daisyui.com/images/stock/photo-1606107557195-0e29a4b5b4aa.webp"
                alt="Shoes"
                className="rounded-xl"
              />
            </figure>
            <div className="card-body items-center text-center">
              <h2 className="card-title">Card Title</h2>
              <p>
                A card component has a figure, a body part, and inside body
                there are title and actions parts
              </p>
              <div className="card-actions">
                <button className="btn btn-primary">Buy Now</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
