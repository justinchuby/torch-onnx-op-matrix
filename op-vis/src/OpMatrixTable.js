import React from 'react';
import GridTable from '@nadavshaar/react-grid-table';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Popover from 'react-bootstrap/Popover';

const CellPopover = React.forwardRef(
  ({ popper, children, show: _, ...props }, ref) => {
    return (
      <Popover ref={ref} placement="right" {...props}>
        <Popover.Header>Sampled Exception Details</Popover.Header>
        <Popover.Body>{children}</Popover.Body>
      </Popover>
    );
  }
);

// TODO: Make the exception details body wider and scrollable.

const ExceptionDetails = ({ exceptions }) => {
  if (exceptions && exceptions.length > 0) {
    return (
      <>
        {exceptions.map((exception, index) => {
          return (
            <div key={index} stype={{ marginTop: '20px' }}>
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
  } else {
    return <p>All tests passed. ✨</p>;
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
  let totalConnt;
  let correctCount;
  let exceptions;

  if (data[column.field]) {
    totalConnt = data[column.field].total_count;
    correctCount = data[column.field].correct_count;
    exceptions = data[column.field].exceptions;
  } else {
    totalConnt = 0;
    correctCount = 0;
    exceptions = [];
  }

  let supportClass;
  if (totalConnt === 0) {
    supportClass = 'support-unknown';
  } else if (correctCount === totalConnt) {
    supportClass = 'support-yes';
  } else if (correctCount === 0) {
    supportClass = 'support-no';
  } else {
    supportClass = 'support-partial';
  }
  const popover = (
    <CellPopover>
      <ExceptionDetails exceptions={exceptions} />
    </CellPopover>
  );
  //  id={`${rowIndex}-${colIndex}`}
  return (
    <div className={`${supportClass}`}>
      <div
        className={'rgt-cell-inner'}
        // style={{ display: 'flex', alignItems: 'center', overflow: 'hidden' }}
      >
        <OverlayTrigger trigger="click" placement="right" overlay={popover}>
          <span className="rgt-text-truncate support-text">
            {correctCount} / {totalConnt}
          </span>
        </OverlayTrigger>
      </div>
    </div>
  );
};

const columns = [
  {
    id: 0,
    field: 'operator',
    label: 'operator',
    editable: false,
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
    width: '75px',
    sortable: false,
    editable: false,
    searchable: false,
  }))
);

const OpMatrixTable = ({ rows }) => (
  <GridTable columns={columns} rows={rows} pageSizes={[20, 100, 1000]} />
);

export default OpMatrixTable;
