import React from "react";
import GridTable from '@nadavshaar/react-grid-table';

// // custom cell component
// const Username = ({ tableManager, value, field, data, column, colIndex, rowIndex }) => {
//     return (
//         <div className='rgt-cell-inner support-partial' style={{display: 'flex', alignItems: 'center', overflow: 'hidden'}}>
//             <img src={data.avatar} alt="user avatar" />
//             <span className='rgt-text-truncate' style={{marginLeft: 10}}>{value}</span>
//         </div>
//     )
// }


const CellRenderer = ({ tableManager, value, field, data, column, colIndex, rowIndex }) => {
    let supportClass;
    if (data.total_count === 0) {
        supportClass = 'support-unknown';
    } else if (data.success_count === data.total_count) {
        supportClass = 'support-yes';
    } else if (data.success_count === 0) {
        supportClass = 'support-no';
    } else {
        supportClass = 'support-partial';
    }
    return (
        <div className={`rgt-cell-inner ${supportClass}`} style={{display: 'flex', alignItems: 'center', overflow: 'hidden'}}>
            <span className='rgt-text-truncate support-text' >{data.success_count} / {data.total_count}</span>
        </div>
    );
}


const repeat = (arr, n) => Array(n).fill(arr).flat();

const rows = repeat([
    {
        "id": 1,
        "success_count": 5,
        "total_count": 10,
    },
    {
        "id": 2,
        "success_count": 10,
        "total_count": 10,
    },
    {
        "id": 3,
        "success_count": 0,
        "total_count": 10,
    },
    {
        "id": 3,
        "success_count": 0,
        "total_count": 0,
    },
], 1000);

const columns = [
    {
        id: 1,
        field: 'status',
        label: 'Status',
        cellRenderer: CellRenderer,
    },
    // {
    //     id: 2,
    //     field: 'gender',
    //     label: 'Gender',
    // },
    // {
    //     id: 3,
    //     field: 'last_visited',
    //     label: 'Last Visited',
    //     sort: ({a, b, isAscending}) => {
    //         let aa = a.split('/').reverse().join(),
    //         bb = b.split('/').reverse().join();
    //         return aa < bb ? isAscending ? -1 : 1 : (aa > bb ? isAscending ? 1 : -1 : 0);
    //     }
    // },
    // {
    //     id: 4,
    //     field: 'test',
    //     label: 'Score',
    //     getValue: ({value, column}) => value.x + value.y
    // }
];

const MyAwesomeTable = () => <GridTable columns={columns} rows={rows} pageSize={1000} />;

export default MyAwesomeTable;