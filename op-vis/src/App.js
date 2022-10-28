import './App.css';
import OpMatrixTable from './Table';
import surveyData from './data/op_survey.json';

// TODO: Pages, sorting, exceptions

const Page = ({ torch_version, onnx_version, opset, test_results }) => {
  return (
    <div className="Page" id={opset} >
      <OpMatrixTable rows={test_results} />
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <header className="App-header">
        {surveyData.map((data, index) => {
          return (
            <Page
              torch_version={data.torch_version}
              onnx_version={data.onnx_version}
              opset={data.opset}
              test_results={data.test_results}
            />
          );
        })}
      </header>
    </div>
  );
}

export default App;
