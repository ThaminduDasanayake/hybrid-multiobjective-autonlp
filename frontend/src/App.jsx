import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import HistoryAnalysis from "./pages/HistoryAnalysis";
import RunAutoML from "./pages/RunAutoML";
import ThesisDefense from "./pages/ThesisDefense";

const App = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<RunAutoML />} />
          <Route path="history" element={<HistoryAnalysis />} />
          <Route path="ablations" element={<ThesisDefense />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};
export default App;
