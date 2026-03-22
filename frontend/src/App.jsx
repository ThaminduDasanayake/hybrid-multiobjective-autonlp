import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import RunAutoML from "./pages/RunAutoML";
import RunHistory from "./pages/RunHistory";
import JobDetail from "./pages/JobDetail";

const App = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<RunAutoML />} />
          <Route path="history" element={<RunHistory />} />
          <Route path="history/:jobId" element={<JobDetail />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};
export default App;
