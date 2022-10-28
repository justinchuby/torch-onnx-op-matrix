import React, { useState, useEffect } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import Container from 'react-bootstrap/Container';
import Spinner from 'react-bootstrap/Spinner';

import OpMatrixTable from './OpMatrixTable';

import './App.css';

const Page = ({ torch_version, onnx_version, opset, test_results }) => {
  return (
    <div className="Page" id={opset}>
      <code>
        Tested on PyTorch {torch_version}
        {onnx_version ? `; ONNX ${onnx_version}` : ''}
      </code>
      <OpMatrixTable rows={test_results} />
    </div>
  );
};

function App() {
  const [data, setData] = useState([]);
  const getData = () => {
    fetch('data/op_survey.json', {
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
    })
      .then(function (response) {
        return response.json();
      })
      .then(function (jsonData) {
        setData(jsonData);
      });
  };
  useEffect(() => {
    getData();
  }, []);
  return (
    <div className="App">
      <Container>
        <h1>torch.onnx Op Support Matrix</h1>
        {data.length === 0 ? (
          <Spinner animation="border" role="status">
            <span className="visually-hidden">Loading data...</span>
          </Spinner>
        ) : (
          <Tabs defaultActiveKey="9" className="mb-3">
            {data.map((data, index) => {
              return (
                <Tab
                  eventKey={`${data.opset}`}
                  title={`${data.opset}`}
                  key={index}
                >
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
        )}
      </Container>
    </div>
  );
}

export default App;
