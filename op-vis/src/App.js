import axios from 'axios';
import React, { useState, useEffect } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import Container from 'react-bootstrap/Container';
import ProgressBar from 'react-bootstrap/ProgressBar';
import Badge from 'react-bootstrap/Badge';

import OpMatrixTable from './OpMatrixTable';

import './App.css';

const Page = ({ torch_version, onnx_version, opset, test_results }) => {
  return (
    <div className="Page" id={opset}>
      <span>
        Tested on <Badge bg="secondary">PyTorch {torch_version}</Badge>
        {onnx_version ? (
          <Badge bg="secondary">{`ONNX ${onnx_version}`}</Badge>
        ) : null}
      </span>
      <OpMatrixTable rows={test_results} />
    </div>
  );
};

function App() {
  const [data, setData] = useState([]);
  const [progress, setProgress] = useState(0);
  const getData = () => {
    axios
      .get('data/op_survey.json', {
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        onDownloadProgress: (progressEvent) => {
          const estimatedTotalSize = 286712384;
          const percentage = Math.round(
            (progressEvent.loaded / estimatedTotalSize) * 100
          );
          setProgress(percentage);
        },
      })
      .then(function (response) {
        setData(response.data);
      })
      .catch(function (error) {
        // handle error
        console.log(error);
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
          <ProgressBar
            animated
            now={progress}
            label="Downloading a few hundred megabytes"
          />
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
