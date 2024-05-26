import axios from 'axios';
import React, { useState, useEffect } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import Container from 'react-bootstrap/Container';
import ProgressBar from 'react-bootstrap/ProgressBar';
import Badge from 'react-bootstrap/Badge';

import OpMatrixTable from './OpMatrixTable';

import './App.scss';

const Page = ({ torch_version, onnx_version, opset, test_results }) => {
  return (
    <div className="Page" id={opset}>
      <div className="pb-3">
        <span>
          Tested on: <Badge pill bg="info">PyTorch {torch_version}</Badge> {' '}
          {onnx_version ? (
            <Badge pill bg="info">{`ONNX ${onnx_version}`}</Badge>
          ) : null}
        </span>
      </div>
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
        <div className='py-2'>
          <h5>
          <b className='mt-2'>Legend:</b>{' '}
          <Badge className="support-yes">Supported: All tests passed</Badge>{' '}
          <Badge className="support-partial">Partial Support: Some tests passed</Badge>{' '}
          <Badge className="support-no">Broken Support: All tests failed</Badge>{' '}
          <Badge className="support-unknown">Unsupported: No tests ran</Badge>{' '}
          </h5>
        </div>
        {data.length === 0 ? (
          <ProgressBar
            animated
            variant="info"
            now={progress}
            label="Downloading a few hundred megabytes"
          />
        ) : (
          <Tabs justify defaultActiveKey="9" className="mb-3">
            {data.map((data, index) => {
              return (
                <Tab
                  eventKey={`${data.dtype}`}
                  title={`${data.dtype}`}
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
