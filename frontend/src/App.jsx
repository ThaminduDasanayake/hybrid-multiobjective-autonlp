import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import HistoryAnalysis from "./pages/HistoryAnalysis";
import RunAutoML from "./pages/RunAutoML";
import Experiments from "./pages/Experiments.jsx";

const App = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<RunAutoML />} />
          <Route path="history" element={<HistoryAnalysis />} />
          <Route path="experiments" element={<Experiments />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};
export default App;
