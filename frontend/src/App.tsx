import { BrowserRouter, Route, Routes } from "react-router-dom";
import Landing from "./pages/Landing";
import Generate from "./pages/Generate";
import Exhibit from "./pages/Exhibit";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/generate" element={<Generate />} />
        <Route path="/exhibit/:id" element={<Exhibit />} />
      </Routes>
    </BrowserRouter>
  );
}
