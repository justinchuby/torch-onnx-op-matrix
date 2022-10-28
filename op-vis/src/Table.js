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
    // <CellPopover id={`${data.operator}-${rowIndex}-${colIndex}`}>
    <CellPopover>
      <ExceptionDetails exceptions={exceptions} />
    </CellPopover>
  );
  //  id={`${rowIndex}-${colIndex}`}
  return (
    <div className={`rgt-cell ${supportClass}`}>
      <div
        className={'rgt-cell-inner'}
        style={{ display: 'flex', alignItems: 'center', overflow: 'hidden' }}
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
    id: 1,
    field: 'operator',
    label: 'operator',
  },
  {
    id: 2,
    field: 'uint8',
    label: 'uint8',
    cellRenderer: CellRenderer,
  },
  {
    id: 3,
    field: 'int8',
    label: 'int8',
    cellRenderer: CellRenderer,
  },
  {
    id: 4,
    field: 'int16',
    label: 'int16',
    cellRenderer: CellRenderer,
  },
  {
    id: 5,
    field: 'int32',
    label: 'int32',
    cellRenderer: CellRenderer,
  },
  {
    id: 6,
    field: 'int64',
    label: 'int64',
    cellRenderer: CellRenderer,
  },
  {
    id: 7,
    field: 'bool',
    label: 'bool',
    cellRenderer: CellRenderer,
  },
  {
    id: 8,
    field: 'float16',
    label: 'float16',
    cellRenderer: CellRenderer,
  },
  {
    id: 9,
    field: 'float32',
    label: 'float32',
    cellRenderer: CellRenderer,
  },
  {
    id: 10,
    field: 'float64',
    label: 'float64',
    cellRenderer: CellRenderer,
  },
  {
    id: 11,
    field: 'bfloat16',
    label: 'bfloat16',
    cellRenderer: CellRenderer,
  },
  {
    id: 12,
    field: 'complex64',
    label: 'complex64',
    cellRenderer: CellRenderer,
  },
  {
    id: 13,
    field: 'complex128',
    label: 'complex128',
    cellRenderer: CellRenderer,
  },
  {
    id: 14,
    field: 'qint8',
    label: 'qint8',
    cellRenderer: CellRenderer,
  },
  {
    id: 15,
    field: 'quint8',
    label: 'quint8',
    cellRenderer: CellRenderer,
  },
];

const OpMatrixTable = ({ rows }) => <GridTable columns={columns} rows={rows} />;

export default OpMatrixTable;
