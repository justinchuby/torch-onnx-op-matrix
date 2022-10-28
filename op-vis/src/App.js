import './App.css';
import OpMatrixTable from './Table';
import surveyData from './data/op_survey.json';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import Container from 'react-bootstrap/Container';


// TODO: Pages, sorting, exceptions

const Page = ({ torch_version, onnx_version, opset, test_results }) => {
  return (
    <div className="Page" id={opset}>
      <code>Tested on PyTorch {torch_version}; ONNX {onnx_version}</code>
      <OpMatrixTable rows={test_results} />
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <Container>
        <h1>torch.onnx Op Support Matrix</h1>
        <Tabs
          defaultActiveKey="9"
          className="mb-3"
        >
          {surveyData.map((data, index) => {
            return (
              <Tab eventKey={`${data.opset}`} title={`${data.opset}`} key={index}>
                <Page
                  torch_version={data.torch_version}
                  onnx_version={data.onnx_version}
                  opset={data.opset}
                  test_results={data.test_results}
                />
              </Tab>
            );
          })}
        </Tabs>
      </Container>
    </div>
  );
}

export default App;
