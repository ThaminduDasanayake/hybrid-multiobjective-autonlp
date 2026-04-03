import { BrowserRouter, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import RunAutoML from "./pages/RunAutoML";
import JobDetail from "./pages/JobDetail";
import HowItWorks from "./pages/HowItWorks";
import FeedbackWidget from "./components/FeedbackWidget";

const App = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/run" element={<RunAutoML />} />
        <Route path="/jobs/:jobId" element={<JobDetail />} />
        <Route path="/how-it-works" element={<HowItWorks />} />
      </Routes>
      <FeedbackWidget />
    </BrowserRouter>
  );
};
export default App;
