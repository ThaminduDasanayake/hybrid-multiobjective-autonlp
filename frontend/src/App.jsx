import { BrowserRouter, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import RunAutoML from "./pages/RunAutoML";
import JobDetail from "./pages/JobDetail";

const App = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/run" element={<RunAutoML />} />
        <Route path="/history/:jobId" element={<JobDetail />} />
      </Routes>
    </BrowserRouter>
  );
};
export default App;
