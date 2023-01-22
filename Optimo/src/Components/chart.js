import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import api from '../services/api';

const Chart = () => {
  const [chartData, setChartData] = useState({});

  useEffect(() => {
    const fetchData = async () => {
      const data = await api.get('/chart-data');
      setChartData(data);
    }
    fetchData();
  }, []);

  return (
    <div>
      <Line
        data={chartData}
        options={{
          maintainAspectRatio: false
        }}
      />
    </div>
  );
}

export default Chart;
