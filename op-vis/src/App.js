import './App.css';
import OpMatrixTable from './Table';
import surveyData from './data/op_survey.json';
import { createHashRouter, RouterProvider, Route } from 'react-router-dom';

// TODO: Pages, sorting, exceptions


const Page = ({ torch_version, onnx_version, opset, test_results }) => {
  return (
    <div className="Page" id={opset}>
      <OpMatrixTable rows={test_results} />
      {/* <p>hello {opset}</p> */}
    </div>
  );
};

const router = createHashRouter(
  surveyData.map((data, index) => {
    return {
      path: `${data.opset}`,
      element: (
        <Page
          torch_version={data.torch_version}
          onnx_version={data.onnx_version}
          opset={data.opset}
          test_results={data.test_results}
        />
      ),
    };
  })
);

function App() {
  return (
    <div className="App">
      <RouterProvider router={router} />
    </div>
  );
}

export default App;
