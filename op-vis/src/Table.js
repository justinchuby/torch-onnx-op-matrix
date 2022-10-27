import React from "react";
import GridTable from '@nadavshaar/react-grid-table';
import opset9Data from './data/op_survey_opset_9.json';


const CellRenderer = ({ tableManager, value, field, data, column, colIndex, rowIndex }) => {
    let totalConnt;
    let correctCount;

    if (data[column.field]) {
        totalConnt = data[column.field].total_count;
        correctCount = data[column.field].correct_count;
    } else {
        totalConnt = 0;
        correctCount = 0;
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
    return (
        <div className={`rgt-cell ${supportClass}`}>
            <div className={'rgt-cell-inner'} style={{display: 'flex', alignItems: 'center', overflow: 'hidden'}}>
                <span className='rgt-text-truncate support-text' >{correctCount} / {totalConnt}</span>
            </div>
        </div>
    );
}


// const repeat = (arr, n) => Array(n).fill(arr).flat();

// const rows = repeat([
//     {
//         "id": 1,
//         "operator": "aten::add",
//         "uint8": {
//             "correct_count": 6,
//             "total_count": 10,
//         },
//         "float32": {
//             "correct_count": 5,
//             "total_count": 10,
//         }
//     },
//     {
//         "id": 2,
//         "operator": "aten::sub",
//         "uint8": {
//             "correct_count": 10,
//             "total_count": 10,
//         },
//         "float32": {
//             "correct_count": 10,
//             "total_count": 10,
//         }
//     },
//     {
//         "id": 3,
//         "operator": "aten::matmul",
//         "uint8": {
//             "correct_count": 0,
//             "total_count": 10,
//         },
//         "float32": {
//             "correct_count": 0,
//             "total_count": 10,
//         }
//     },
//     {
//         "id": 3,
//         "operator": "aten::conv2d",
//         "uint8": {
//             "correct_count": 0,
//             "total_count": 0,
//         },
//         "float32": {
//             "correct_count": 5,
//             "total_count": 10,
//         }
//     },
// ], 1000);

const rows = opset9Data;

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
        field: 'qint8',
        label: 'qint8',
        cellRenderer: CellRenderer,
    },
    {
        id: 13,
        field: 'quint8',
        label: 'quint8',
        cellRenderer: CellRenderer,
    },
    {
        id: 14,
        field: 'complex64',
        label: 'complex64',
        cellRenderer: CellRenderer,
    },
    {
        id: 15,
        field: 'complex128',
        label: 'complex128',
        cellRenderer: CellRenderer,
    },
];

const OpMatrixTable = () => <GridTable columns={columns} rows={rows} pageSize={1000} />;

export default OpMatrixTable;
