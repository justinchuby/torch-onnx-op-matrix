import React from 'react';
import GridTable from '@nadavshaar/react-grid-table';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Popover from 'react-bootstrap/Popover';

const CellPopover = React.forwardRef(
  ({ popper, children, show: _, ...props }, ref) => {
    return (
      <Popover ref={ref} body {...props}>
        {children}
      </Popover>
    );
  }
);

const ExceptionDetails = ({ exceptions }) => {
  //   <div>
  <h3>Exception Details</h3>;
  {
    /* <div>
      {exceptions.map((exception, index) => {
        return <p key={index}>{exception.message}</p>;
      })}
    </div>
  </div>; */
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
        style={{ display: 'flex', alignItems: 'center', overflow: 'hidden' }}
      >
        <OverlayTrigger trigger="click" placement="right" overlay={popover}>
          <span className="rgt-text-truncate support-text">
            {correctCount} / {totalConnt} &nbsp;&nbsp;&nbsp;
          </span>
        </OverlayTrigger>
      </div>
    </div>
  );
};

const columns = [
  {
    id: 1,
    field: 'operator',
    label: 'operator',
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
  }))
);

const OpMatrixTable = ({ rows }) => <GridTable columns={columns} rows={rows} pageSizes={[30, 100, 1000]} />;

export default OpMatrixTable;
