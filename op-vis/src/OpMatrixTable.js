import React from 'react';
import GridTable from '@nadavshaar/react-grid-table';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Popover from 'react-bootstrap/Popover';

const CellPopover = React.forwardRef(
  ({ popper, children, show: _, ...props }, ref) => {
    return (
      <Popover
        ref={ref}
        placement="right"
        style={{ width: '300px' }}
        {...props}
      >
        <Popover.Header>Test Details</Popover.Header>
        <Popover.Body>{children}</Popover.Body>
      </Popover>
    );
  }
);

const TorchOp = ({ operator }) => {
  return (
    <a
      href={`https://pytorch.org/docs/stable/generated/torch.${operator}.html`}
      target="_blank"
      rel="noopener noreferrer"
    >
      <code>torch.{operator}</code>
    </a>
  );
};

const ExceptionDetails = ({
  exceptions,
  operator,
  dtype,
  correctCount,
  totalCount,
}) => {
  if (exceptions && exceptions.length > 0) {
    return (
      <>
        <p>
          Tested <TorchOp operator={operator} /> with <code>{dtype}</code>{' '}
          inputs.{' '}
          <code>
            {correctCount} / {totalCount}
          </code>{' '}
          passed.
        </p>
        <p>Sampled failures:</p>
        {exceptions.map((exception, index) => {
          return (
            <div key={index}>
              <b>
                #{index + 1}) {exception.type}
              </b>
              <p style={{ whiteSpace: 'pre-line' }}>{exception.message}</p>
              <span>Inputs: </span>
              <pre>{exception.inputs}</pre>
              <span>kwargs: </span>
              <pre>{exception.kwargs}</pre>
              <pre>{exception.traceback}</pre>
            </div>
          );
        })}
      </>
    );
  } else if (totalCount === 0) {
    return (
      <p>
        No tests were run for <TorchOp operator={operator} /> with{' '}
        <code>{dtype}</code> inputs.
      </p>
    );
  } else {
    return (
      <p>
        All <code>{totalCount}</code> tests passed for{' '}
        <TorchOp operator={operator} /> with <code>{dtype}</code> inputs.
      </p>
    );
  }
};

const CellRenderer = ({
  tableManager,
  value,
  field,
  data,
  column,
  colIndex,
  rowIndex,
}) => {
  let totalCount;
  let correctCount;
  let exceptions;

  if (data[column.field]) {
    totalCount = data[column.field].total_count;
    correctCount = data[column.field].correct_count;
    exceptions = data[column.field].exceptions;
  } else {
    totalCount = 0;
    correctCount = 0;
    exceptions = [];
  }

  let supportClass;
  if (totalCount === 0) {
    supportClass = 'support-unknown';
  } else if (correctCount === totalCount) {
    supportClass = 'support-yes';
  } else if (correctCount === 0) {
    supportClass = 'support-no';
  } else {
    supportClass = 'support-partial';
  }
  const popover = (
    <CellPopover>
      <ExceptionDetails
        exceptions={exceptions}
        operator={data.operator}
        dtype={column.field}
        correctCount={correctCount}
        totalCount={totalCount}
      />
    </CellPopover>
  );
  return (
    <div
      className={`rgt-cell-inner ${supportClass}`}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        margin: '0.5px',
        paddingTop: '10px',
        paddingBottom: '10px',
        textAlign: 'center',
      }}
    >
      <OverlayTrigger trigger="click" placement="right" overlay={popover}>
        <span
          className="rgt-text-truncate support-text"
          style={{ cursor: 'help', fontSize: '0.85em' }}
        >
          <b>{correctCount} / {totalCount}</b>
        </span>
      </OverlayTrigger>
    </div>
  );
};

const columns = [
  {
    id: 0,
    field: 'operator',
    label: 'operator',
    editable: false,
    pinned: true,
  },
];

columns.push(
  ...[
    'uint8',
    'int8',
    'int16',
    'int32',
    'int64',
    'bool',
    'float16',
    'float32',
    'float64',
    'bfloat16',
    'complex64',
    'complex128',
    'qint8',
    'quint8',
  ].map((dtype, index) => ({
    id: index + 1,
    field: dtype,
    label: dtype,
    cellRenderer: CellRenderer,
    width: 'max-content',
    sortable: false,
    editable: false,
    searchable: false,
  }))
);

const OpMatrixTable = ({ rows }) => (
  <GridTable
    columns={columns}
    rows={rows}
    pageSizes={[20, 100, 1000]}
    minColumnResizeWidth={40}
  />
);

export default OpMatrixTable;
